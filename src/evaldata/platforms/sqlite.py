"""`SqliteAdapter`: in-process SQLite execution backend over the stdlib `sqlite3` driver."""

import sqlite3
import time
from types import TracebackType
from typing import Self

from evaldata.platforms.base import execution_error, untyped_rows_or_error
from evaldata.types import ExecutionResult


class SqliteAdapter:
    """Executes SQL against an in-process SQLite database via the stdlib driver."""

    def __init__(self, database: str = ":memory:") -> None:
        """Open a SQLite connection to `database` (default in-memory).

        Args:
            database: A filesystem path to a SQLite database, `:memory:` for an in-process
                database, or a `file:` URI (e.g. a shared-cache in-memory database, which lives
                while at least one connection to it stays open).
        """
        # check_same_thread=False: cancel() calls interrupt() from another thread mid-execute().
        # isolation_level=None: autocommit, so writes are durable instead of rolled back on close().
        self._conn = sqlite3.connect(
            database, uri=database.startswith("file:"), check_same_thread=False, isolation_level=None
        )

    def cancel(self) -> None:
        """Interrupt the query currently executing on this connection.

        Safe to call from another thread while `execute` is blocked, and a no-op when no
        query is running. The interrupted `execute` raises `sqlite3.OperationalError`, which
        it surfaces as `ExecutionResult.error` like any other query failure.
        """
        self._conn.interrupt()

    def close(self) -> None:
        """Release the underlying SQLite connection."""
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
            An `ExecutionResult` with the returned rows, an `UntypedSchema`, and latency:
            SQLite is dynamically typed and the driver reports no result-column types. Query
            failures are returned as `ExecutionResult.error` rather than raised.
        """
        start = time.perf_counter()
        try:
            cursor = self._conn.execute(sql)
            description = cursor.description or []
            rows_raw = cursor.fetchall()
        except sqlite3.Error as e:
            elapsed = time.perf_counter() - start
            return ExecutionResult(rows=[], schema=None, latency_seconds=elapsed, error=execution_error(e))
        elapsed = time.perf_counter() - start
        return untyped_rows_or_error([desc[0] for desc in description], rows_raw, elapsed)
