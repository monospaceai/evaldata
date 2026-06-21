"""`DuckDBAdapter`: in-process DuckDB execution backend."""

import time
from types import TracebackType
from typing import Self

import duckdb

from evaldata.platforms.base import execution_error, rows_or_error
from evaldata.types import Column, ExecutionResult, SqlType


class DuckDBAdapter:
    """Executes SQL against an in-process DuckDB database."""

    def __init__(self, database: str = ":memory:") -> None:
        """Open a DuckDB connection to `database` (default `:memory:`)."""
        self._conn = duckdb.connect(database)

    def cancel(self) -> None:
        """Interrupt the query currently executing on this connection.

        Safe to call from another thread while `execute` is blocked, and a no-op when no
        query is running. The interrupted `execute` raises `duckdb.InterruptException`,
        which it surfaces as `ExecutionResult.error` like any other query failure.
        """
        self._conn.interrupt()

    def close(self) -> None:
        """Release the underlying DuckDB connection (file handle / WAL lock).

        Explicit close matters on Windows, where WAL locks make implicit cleanup unreliable.
        """
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
            cursor = self._conn.execute(sql)
            description = cursor.description or []
            rows_raw = cursor.fetchall()
        except duckdb.Error as e:
            elapsed = time.perf_counter() - start
            return ExecutionResult(rows=[], schema=None, latency_seconds=elapsed, error=execution_error(e))
        elapsed = time.perf_counter() - start
        columns: list[Column] = []
        for desc in description:
            name, type_ = desc[0], desc[1]
            null_ok = desc[6] if len(desc) > 6 else None
            columns.append(Column(name=name, type=SqlType.parse(str(type_), "duckdb"), nullable=null_ok))
        return rows_or_error(columns, rows_raw, elapsed)
