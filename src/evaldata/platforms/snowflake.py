"""`SnowflakeAdapter`: Snowflake execution backend over `snowflake-connector-python`."""

import contextlib
import math
import time
from types import TracebackType
from typing import Self

import snowflake.connector
from snowflake.connector.constants import FIELD_ID_TO_NAME
from snowflake.connector.cursor import ResultMetadata, SnowflakeCursor

from evaldata.platforms.base import execution_error, rows_or_error
from evaldata.types import Column, ExecutionError, ExecutionFailure, ExecutionResult, ExecutionSuccess, SqlType


def _type_string(field_name: str, precision: int | None, scale: int | None, internal_size: int | None) -> str:
    """Render a Snowflake internal field-type name as a SQL type string.

    Args:
        field_name: The field-type name from `FIELD_ID_TO_NAME`, e.g. `"FIXED"`, `"TEXT"`.
        precision: The column's precision, when reported.
        scale: The column's scale, when reported.
        internal_size: The column's internal size (characters/bytes), when reported.

    Returns:
        A SQL type string parseable by SQLGlot's snowflake dialect, e.g. `"NUMBER(38,0)"`,
        `"VARCHAR(16777216)"`. Unrecognized field names are returned unchanged.
    """
    match field_name:
        case "FIXED":
            return f"NUMBER({precision},{scale})" if precision is not None else "NUMBER"
        case "REAL":
            return "FLOAT"
        case "TEXT":
            return f"VARCHAR({internal_size})" if internal_size is not None else "VARCHAR"
        case "BINARY":
            return f"BINARY({internal_size})" if internal_size is not None else "BINARY"
        case "BOOLEAN":
            return "BOOLEAN"
        case "DATE":
            return "DATE"
        case "TIME":
            return f"TIME({scale})" if scale is not None else "TIME"
        case "TIMESTAMP_NTZ" | "TIMESTAMP_LTZ" | "TIMESTAMP_TZ":
            return f"{field_name}({scale})" if scale is not None else field_name
        case "TIMESTAMP":
            return "TIMESTAMP"
        case "VARIANT":
            return "VARIANT"
        case "OBJECT":
            return "OBJECT"
        case "ARRAY":
            return "ARRAY"
        case "GEOGRAPHY":
            return "GEOGRAPHY"
        case "GEOMETRY":
            return "GEOMETRY"
        case _:
            return field_name


def _column_from_metadata(meta: ResultMetadata) -> Column:
    """Build a `Column` from one `cursor.description` entry.

    Args:
        meta: The result-column metadata reported by the driver.

    Returns:
        A `Column` with a `SqlType` parsed in the snowflake dialect.
    """
    field_name = FIELD_ID_TO_NAME[meta.type_code]
    raw = _type_string(field_name, meta.precision, meta.scale, meta.internal_size)
    return Column(name=meta.name, type=SqlType.parse(raw, "snowflake"), nullable=meta.is_nullable)


class SnowflakeAdapter:
    """Executes SQL against Snowflake via snowflake-connector-python."""

    def __init__(
        self,
        *,
        account: str,
        user: str | None = None,
        password: str | None = None,
        warehouse: str | None = None,
        role: str | None = None,
        database: str | None = None,
        schema: str | None = None,
        authenticator: str | None = None,
        token: str | None = None,
        private_key_file: str | None = None,
        private_key_file_pwd: str | None = None,
        workload_identity_provider: str | None = None,
    ) -> None:
        """Open a Snowflake connection.

        Args:
            account: The Snowflake account identifier.
            user: The Snowflake user name, or `None` to rely on `authenticator`.
            password: The user's password, or `None` when authenticating another way.
            warehouse: The default warehouse, or `None` to leave the session default.
            role: The default role, or `None` to leave the session default.
            database: The default database, or `None` to leave the session default.
            schema: The default schema, or `None` to leave the session default.
            authenticator: The authenticator to use (e.g. `"externalbrowser"`, `"oauth"`),
                or `None` for the connector's default.
            token: An OAuth/token credential, or `None` when not using token auth.
            private_key_file: Path to a PEM-encoded PKCS#8 private key for key-pair auth,
                or `None`.
            private_key_file_pwd: Passphrase for an encrypted `private_key_file`, or `None`
                when the key is unencrypted or unused.
            workload_identity_provider: The workload identity provider, or `None` when not
                using workload identity federation.
        """
        connect_kwargs = {
            "account": account,
            "user": user,
            "password": password,
            "warehouse": warehouse,
            "role": role,
            "database": database,
            "schema": schema,
            "authenticator": authenticator,
            "token": token,
            "private_key_file": private_key_file,
            "private_key_file_pwd": private_key_file_pwd,
            "workload_identity_provider": workload_identity_provider,
        }
        self._conn = snowflake.connector.connect(**{k: v for k, v in connect_kwargs.items() if v is not None})
        self._cursor: SnowflakeCursor | None = None

    @property
    def connection(self) -> snowflake.connector.SnowflakeConnection:
        """The live Snowflake connection backing this adapter."""
        return self._conn

    def cancel(self) -> None:
        """Abort the query currently executing on this connection, if any.

        Safe to call from another thread while `execute` is blocked; best-effort, so all
        failures are swallowed.
        """
        cursor = self._cursor
        if cursor is not None and cursor.sfqid:
            with contextlib.suppress(Exception):
                cursor.abort_query(cursor.sfqid)

    def close(self) -> None:
        """Release the underlying Snowflake connection."""
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
        """Execute one SQL statement against Snowflake.

        Args:
            sql: The SQL statement to execute.

        Returns:
            An `ExecutionResult` with the returned rows, schema, and latency. Query
            failures are returned as an `ExecutionFailure` rather than raised.
        """
        return self._execute(sql)

    def execute_with_timeout(self, sql: str, timeout_seconds: float) -> ExecutionResult:
        """Execute one statement with the connector's native timeout.

        Args:
            sql: The SQL statement to execute.
            timeout_seconds: Positive statement deadline in seconds.

        Returns:
            The structured statement result.
        """
        return self._execute(sql, timeout=math.ceil(timeout_seconds))

    def ping(self) -> bool:
        """Return whether Snowflake reports this session as valid."""
        try:
            return bool(self._conn.is_valid())
        except Exception:
            return False

    def is_disconnect(self, error: ExecutionError) -> bool:
        """Return whether a connection-class error leaves this session unusable."""
        if error.sqlstate is not None and error.sqlstate.startswith("08"):
            return True
        cause = error.cause
        connection_errors = (
            snowflake.connector.errors.InterfaceError,
            snowflake.connector.errors.OperationalError,
            snowflake.connector.errors.RequestTimeoutError,
            snowflake.connector.errors.ServiceUnavailableError,
        )
        return isinstance(cause, connection_errors)

    def _execute(self, sql: str, *, timeout: int | None = None) -> ExecutionResult:
        """Execute user SQL once, optionally using the connector deadline.

        Returns:
            The structured statement result.
        """
        start = time.perf_counter()
        cursor = self._conn.cursor()
        self._cursor = cursor
        try:
            if timeout is None:
                cursor.execute(sql)
            else:
                cursor.execute(sql, timeout=timeout)
            description = cursor.description
            rows_raw = cursor.fetchall() if description is not None else []
        except Exception as e:  # noqa: BLE001 - execute must never raise; failures return as ExecutionFailure
            elapsed = time.perf_counter() - start
            return ExecutionFailure(latency_seconds=elapsed, error=execution_error(e))
        finally:
            self._cursor = None
            with contextlib.suppress(Exception):
                cursor.close()
        elapsed = time.perf_counter() - start
        if description is None:
            return ExecutionSuccess(rows=[], schema=None, latency_seconds=elapsed)
        columns = [_column_from_metadata(meta) for meta in description]
        return rows_or_error(columns, [tuple(row) for row in rows_raw], elapsed)
