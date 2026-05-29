"""``DuckDBAdapter``: in-process DuckDB execution backend.

Conforms to ``PlatformAdapter``. Uses ``duckdb`` directly (no SQLAlchemy) and
reports native DuckDB type strings via the cursor's ``description``: each
``description[i][1]`` is a ``DuckDBPyType`` whose ``str()`` yields the type
SQLGlot's ``duckdb`` dialect parses (``INTEGER``, ``STRUCT(a INTEGER, b VARCHAR)``,
``INTEGER[]``, ...). All query failures surface as ``duckdb.Error`` and are
returned via ``ExecutionResult.error`` rather than raised.

Connection lifecycle is owned by the adapter: ``close()`` and the context-manager
protocol (``__enter__`` / ``__exit__``) release the underlying handle. Documented
for DuckDB (issue #3573) and required on Windows where WAL file locks (issue #1365)
make ``del``-based cleanup unreliable. These are NOT on the ``PlatformAdapter``
Protocol — adapters may offer them as a convention.
"""

import time
from types import TracebackType
from typing import Self

import duckdb

from data_eval.types import Column, ExecutionResult


class DuckDBAdapter:
    """Executes SQL against an in-process DuckDB database."""

    def __init__(self, database: str = ":memory:") -> None:
        """Open a DuckDB connection to ``database`` (default ``:memory:``)."""
        self._conn = duckdb.connect(database)

    def close(self) -> None:
        """Release the underlying DuckDB connection (file handle / WAL lock)."""
        self._conn.close()

    def __enter__(self) -> Self:
        """Return self; the connection is already open from ``__init__``."""
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
        """Execute one SQL statement; return rows + schema + latency, or error-as-value."""
        start = time.perf_counter()
        try:
            cursor = self._conn.execute(sql)
            description = cursor.description or []
            rows_raw = cursor.fetchall()
        except duckdb.Error as e:
            elapsed = time.perf_counter() - start
            return ExecutionResult(
                rows=[],
                schema=None,
                latency_seconds=elapsed,
                error=str(e),
            )
        elapsed = time.perf_counter() - start
        # Single pass over description; tuple-unpack so a malformed PEP-249
        # entry fails loudly here rather than producing silently-wrong rows.
        schema: list[Column] = []
        names: list[str] = []
        for name, type_, *_ in description:
            schema.append(Column(name=name, type=str(type_)))
            names.append(name)
        rows = [dict(zip(names, row, strict=True)) for row in rows_raw]
        return ExecutionResult(rows=rows, schema=schema, latency_seconds=elapsed)
