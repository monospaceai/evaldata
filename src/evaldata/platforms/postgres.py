"""`PostgresAdapter`: PostgreSQL execution backend over `psycopg` (v3)."""

import contextlib
import math
import time
from types import TracebackType
from typing import Self

import psycopg

from evaldata.platforms.base import execution_error, rows_or_error
from evaldata.types import Column, ExecutionError, ExecutionFailure, ExecutionResult, ExecutionSuccess, SqlType


class PostgresAdapter:
    """Executes SQL against a PostgreSQL database via psycopg (v3)."""

    def __init__(self, conninfo: str = "") -> None:
        """Open a psycopg connection.

        `conninfo` is a libpq connection string — keyword/value
        (`"host=... port=... user=... password=... dbname=..."`) or a
        `postgresql://` URI. Empty uses libpq defaults / `PG*` env vars.
        """
        # A failed statement must not leave later calls in an aborted transaction.
        self._conn = psycopg.connect(conninfo, autocommit=True)
        self._reusable = True

    def cancel(self) -> None:
        """Cancel the current query without raising cancellation errors."""
        with contextlib.suppress(psycopg.Error):  # best-effort; a failed cancel is non-fatal
            self._conn.cancel_safe()

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
            failures are returned as an `ExecutionFailure` rather than raised.
        """
        return self._execute(sql)

    def execute_with_timeout(self, sql: str, timeout_seconds: float) -> ExecutionResult:
        """Execute one statement with PostgreSQL's native statement timeout.

        Args:
            sql: The SQL statement to execute.
            timeout_seconds: Positive statement deadline in seconds.

        Returns:
            The statement result, mapping SQLSTATE `57014` to `budget_exceeded`.
        """
        return self._execute(sql, timeout_milliseconds=math.ceil(timeout_seconds * 1000))

    def is_reusable(self) -> bool:
        """Return whether local timeout-setting cleanup has completed successfully."""
        return getattr(self, "_reusable", True)

    def ping(self) -> bool:
        """Return whether this PostgreSQL session can execute a simple statement."""
        if getattr(self._conn, "closed", False) or getattr(self._conn, "broken", False):
            return False
        try:
            with self._conn.cursor() as cursor:
                cursor.execute("SELECT 1")
            return True
        except psycopg.Error:
            return False

    def is_disconnect(self, error: ExecutionError) -> bool:
        """Return whether a structured error or connection state proves disconnection."""
        return bool(
            (error.sqlstate is not None and error.sqlstate.startswith("08"))
            or getattr(self._conn, "closed", False)
            or getattr(self._conn, "broken", False)
        )

    def _execute(self, sql: str, *, timeout_milliseconds: int | None = None) -> ExecutionResult:
        """Execute user SQL once, optionally restoring a temporary statement timeout.

        Returns:
            The structured statement result.
        """
        start = time.perf_counter()
        previous_timeout: str | None = None
        description = None
        rows_raw = []
        try:
            with self._conn.cursor() as cursor:
                if timeout_milliseconds is not None:
                    cursor.execute("SELECT current_setting('statement_timeout')")
                    current_timeout = cursor.fetchone()
                    if current_timeout is None:
                        self._reusable = False
                        elapsed = time.perf_counter() - start
                        return ExecutionFailure(
                            latency_seconds=elapsed,
                            error=ExecutionError(
                                kind="query_failed", message="statement timeout setting was unavailable"
                            ),
                        )
                    previous_timeout = str(current_timeout[0])
                    cursor.execute("SELECT set_config('statement_timeout', %s, false)", (str(timeout_milliseconds),))
                cursor.execute(sql)  # ty: ignore[no-matching-overload]
                description = cursor.description
                rows_raw = cursor.fetchall() if description is not None else []
        except psycopg.Error as e:
            elapsed = time.perf_counter() - start
            kind = (
                "budget_exceeded"
                if timeout_milliseconds is not None and getattr(e, "sqlstate", None) == "57014"
                else "query_failed"
            )
            return ExecutionFailure(latency_seconds=elapsed, error=execution_error(e, kind))
        finally:
            if previous_timeout is not None:
                try:
                    with self._conn.cursor() as cursor:
                        cursor.execute("SELECT set_config('statement_timeout', %s, false)", (previous_timeout,))
                except psycopg.Error:
                    self._reusable = False
        elapsed = time.perf_counter() - start
        if description is None:
            return ExecutionSuccess(rows=[], schema=None, latency_seconds=elapsed)
        columns = [
            Column(name=col.name, type=SqlType.parse(col.type_display, "postgres"), nullable=col.null_ok)
            for col in description
        ]
        return rows_or_error(columns, rows_raw, elapsed)
