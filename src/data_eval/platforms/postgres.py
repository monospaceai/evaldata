"""`PostgresAdapter`: PostgreSQL execution backend over `psycopg` (v3)."""

import time
from types import TracebackType
from typing import Self

import psycopg

from data_eval.platforms.base import rows_or_error
from data_eval.types import Column, ExecutionResult, SqlType


class PostgresAdapter:
    """Executes SQL against a PostgreSQL database via psycopg (v3)."""

    def __init__(self, conninfo: str = "") -> None:
        """Open a psycopg connection.

        `conninfo` is a libpq connection string — keyword/value
        (`"host=... port=... user=... password=... dbname=..."`) or a
        `postgresql://` URI. Empty uses libpq defaults / `PG*` env vars.
        """
        # autocommit so a failed statement can't poison later calls with an aborted
        # transaction; psycopg's connection context manager is intentionally unused.
        self._conn = psycopg.connect(conninfo, autocommit=True)

    def close(self) -> None:
        """Release the underlying psycopg connection."""
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
        """Execute one SQL statement against the database.

        Args:
            sql: The SQL statement to execute.

        Returns:
            An `ExecutionResult` with the returned rows, schema, and latency. Query
            failures are returned as `ExecutionResult.error` rather than raised.
        """
        start = time.perf_counter()
        try:
            with self._conn.cursor() as cursor:
                # psycopg types `execute` to accept only LiteralString, to steer callers
                # toward parameterized queries. Executing arbitrary caller-provided SQL is
                # this adapter's entire purpose, so that guard is deliberately bypassed.
                cursor.execute(sql)  # ty: ignore[no-matching-overload]
                description = cursor.description
                rows_raw = cursor.fetchall() if description is not None else []
        except psycopg.Error as e:
            elapsed = time.perf_counter() - start
            return ExecutionResult(rows=[], schema=None, latency_seconds=elapsed, error=str(e))
        elapsed = time.perf_counter() - start
        if description is None:
            # Non-row-returning statement (DDL/DML): success, no schema.
            return ExecutionResult(rows=[], schema=None, latency_seconds=elapsed)
        columns = [
            Column(name=col.name, type=SqlType.parse(col.type_display, "postgres"), nullable=col.null_ok)
            for col in description
        ]
        return rows_or_error(columns, rows_raw, elapsed)
