"""`DatabricksAdapter`: Databricks SQL Warehouse execution backend over `databricks-sql-connector`."""

import contextlib
import time
from types import TracebackType
from typing import Any, Self

import databricks.sql
from databricks.sdk.core import Config

from evaldata.platforms.base import execution_error, rows_or_error
from evaldata.types import Column, ExecutionError, ExecutionResult, SqlType


class DatabricksAdapter:
    """Executes SQL against a Databricks SQL Warehouse via databricks-sql-connector."""

    def __init__(
        self,
        *,
        server_hostname: str,
        http_path: str,
        catalog: str | None = None,
        schema: str | None = None,
    ) -> None:
        """Open a Databricks SQL connection.

        Credentials are not passed here.

        Args:
            server_hostname: The workspace hostname (no scheme), e.g. `dbc-xxxx.cloud.databricks.com`.
            http_path: The SQL Warehouse HTTP path, e.g. `/sql/1.0/warehouses/<id>`.
            catalog: The default catalog, or `None` to leave the session default.
            schema: The default schema, or `None` to leave the session default.
        """
        cfg = Config(host=f"https://{server_hostname}")
        connect_kwargs = {"catalog": catalog, "schema": schema}
        self._conn = databricks.sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            credentials_provider=lambda: cfg.authenticate,
            **{k: v for k, v in connect_kwargs.items() if v is not None},
        )
        self._cursor: databricks.sql.client.Cursor | None = None

    def cancel(self) -> None:
        """Cancel the query currently executing on this connection, if any.

        Safe to call from another thread while `execute` is blocked; best-effort, so all
        failures are swallowed.
        """
        cursor = self._cursor
        if cursor is not None:
            with contextlib.suppress(Exception):
                cursor.cancel()

    def close(self) -> None:
        """Release the underlying Databricks connection."""
        self._conn.close()

    def __enter__(self) -> Self:
        """Return self; the connection is already open from `__init__`."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the underlying connection on context-manager exit."""
        self.close()

    def execute(self, sql: str) -> ExecutionResult:
        """Execute one SQL statement against the warehouse.

        Args:
            sql: The SQL statement to execute.

        Returns:
            An `ExecutionResult` with the returned rows, schema, and latency. Query
            failures are returned as `ExecutionResult.error` rather than raised.
        """
        start = time.perf_counter()
        cursor = self._conn.cursor()
        self._cursor = cursor
        try:
            cursor.execute(sql)
            description = cursor.description
            # A duplicate-name result is unrepresentable as name-keyed rows, and the
            # connector's Arrow fetch raises a cryptic error on it — so detect it from the
            # description and skip the fetch, letting `rows_or_error` surface the uniform
            # duplicate error below instead.
            names = [col[0] for col in description] if description is not None else []
            has_duplicates = len(names) != len(set(names))
            rows_raw = cursor.fetchall() if description is not None and not has_duplicates else []
        except Exception as e:  # noqa: BLE001 - execute must never raise; failures return as ExecutionResult.error
            elapsed = time.perf_counter() - start
            return ExecutionResult(rows=[], schema=None, latency_seconds=elapsed, error=execution_error(e))
        finally:
            self._cursor = None
            with contextlib.suppress(Exception):
                cursor.close()
        elapsed = time.perf_counter() - start
        if description is None:
            # Non-row-returning statement (DDL/DML): success, no schema.
            return ExecutionResult(rows=[], schema=None, latency_seconds=elapsed)
        columns = [
            Column(name=name, type=SqlType.parse(type_code, "databricks"), nullable=None)
            for (name, type_code, *_rest) in description
        ]
        return rows_or_error(columns, [tuple(row) for row in rows_raw], elapsed)

    @staticmethod
    def type_probe_sql(sql: str) -> str:
        """Build the `DESCRIBE QUERY` probe that recovers precise types for `sql`.

        `cursor.description` drops parametric type parameters (a `DECIMAL` scale, an `ARRAY`
        element type); `DESCRIBE QUERY` reports the precise `data_type` per column without
        running the query.

        Args:
            sql: The statement whose projected column types to resolve.

        Returns:
            A `DESCRIBE QUERY` statement wrapping `sql` (trailing `;` stripped, which it rejects).
        """
        stripped = sql.rstrip("; \t\n\r")
        return f"DESCRIBE QUERY {stripped}"

    @staticmethod
    def types_from_probe(rows: list[dict[str, Any]]) -> list[SqlType] | ExecutionError:
        """Parse `DESCRIBE QUERY` rows into precise `SqlType`s, in projection order.

        Args:
            rows: The rows returned by executing `type_probe_sql`, in projection order.

        Returns:
            One precise `SqlType` per projected column, in order, or an `ExecutionError` when
            the probe returns no rows or a row without a `data_type`.
        """
        if not rows:
            return ExecutionError(kind="type_probe_failed", message="DESCRIBE QUERY returned no rows")
        # `DESCRIBE QUERY` yields one row per projected column, in projection order.
        types: list[SqlType] = []
        for row in rows:
            data_type = row.get("data_type")
            if not data_type:
                return ExecutionError(
                    kind="type_probe_failed", message=f"DESCRIBE QUERY row missing data_type: {row!r}"
                )
            types.append(SqlType.parse(str(data_type), "databricks"))
        return types
