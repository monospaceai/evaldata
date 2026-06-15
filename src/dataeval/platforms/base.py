"""`PlatformAdapter` Protocol: the contract every platform integration implements."""

import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Protocol, runtime_checkable

from dataeval.types import Column, ExecutionResult, Schema


@runtime_checkable
class PlatformAdapter(Protocol):
    """Executes SQL against a data platform; returns rows + schema + latency.

    Required behavior:
        * On success: return `ExecutionResult` with `rows` populated, `schema_`
          populated (each `Column.type` is a `SqlType` built from the driver's
          native type string parsed in the adapter's static dialect), non-negative
          `latency_seconds`, and `error is None`.
        * On query failure: return `ExecutionResult` with `rows=[]`,
          `schema_=None`, a non-empty `error` string describing the failure,
          and non-negative `latency_seconds`. **Do NOT raise.** (Errors-as-values.)
        * `latency_seconds` measures wall-clock time spent inside `execute()`.
    """

    def execute(self, sql: str) -> ExecutionResult:
        """Execute one SQL statement and return its structured result."""
        ...

    def cancel(self) -> None:
        """Abort the query currently executing on this connection, if any.

        Called from another thread when an `execute` overruns its time budget, so it must
        be safe to invoke while `execute` is blocked. A no-op when no query is running, and
        best-effort: it must not raise — a cancellation that fails or arrives late is
        non-fatal (the overrun is still surfaced as `ExecutionResult.error`).
        """
        ...

    def close(self) -> None:
        """Release the underlying connection/resources. Called once at session end."""
        ...


def execute_within_budget(adapter: PlatformAdapter, sql: str, max_seconds: float | None) -> ExecutionResult:
    """Execute `sql` on `adapter`, cancelling it if it exceeds `max_seconds`.

    With `max_seconds` unset, runs `adapter.execute` directly. Otherwise runs it on a worker
    thread and waits up to `max_seconds`; if the query is still running, calls
    `adapter.cancel()` to abort it and returns an `ExecutionResult.error` describing the
    overrun (errors-as-values — a budget overrun is a failure value, not an exception). The
    cancelled query is awaited so its connection is released before returning.

    Args:
        adapter: The platform adapter to execute against.
        sql: The SQL statement to execute.
        max_seconds: Wall-clock ceiling for the query, or `None` for no limit.

    Returns:
        The adapter's `ExecutionResult` if the query finishes within budget, otherwise an
        `ExecutionResult` with `error` set and `latency_seconds` measuring the elapsed time.
    """
    if max_seconds is None:
        return adapter.execute(sql)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.execute, sql)
        try:
            return future.result(timeout=max_seconds)
        except FuturesTimeout:
            adapter.cancel()
    elapsed = time.perf_counter() - start
    return ExecutionResult(
        rows=[],
        schema=None,
        latency_seconds=elapsed,
        error=f"exceeded cost budget: query did not complete within {max_seconds}s",
    )


def rows_or_error(columns: list[Column], rows_raw: list[tuple[Any, ...]], latency_seconds: float) -> ExecutionResult:
    """Build a successful `ExecutionResult`, or an error one if column names collide.

    Rows are keyed by name (`dict(zip(...))`), which cannot represent two columns sharing
    a name — the later value would silently overwrite the earlier. Rather than lose data,
    duplicate output column names are surfaced as `ExecutionResult.error`.

    Args:
        columns: The result-set columns, in order, as reported by the driver.
        rows_raw: The positional row tuples, aligned with `columns`.
        latency_seconds: Wall-clock time spent executing the query.

    Returns:
        An `ExecutionResult` with name-keyed rows and schema, or `error` set (and no rows
        or schema) when one or more column names are duplicated.
    """
    schema = Schema(root=columns)
    names = schema.names
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        listed = ", ".join(repr(name) for name in duplicates)
        return ExecutionResult(
            rows=[],
            schema=None,
            latency_seconds=latency_seconds,
            error=f"duplicate output column name(s): {listed}",
        )
    rows = [dict(zip(names, row, strict=True)) for row in rows_raw]
    return ExecutionResult(rows=rows, schema=schema, latency_seconds=latency_seconds)
