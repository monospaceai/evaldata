"""`PlatformAdapter` Protocol: the contract every platform integration implements."""

from collections import Counter
from typing import Any, Protocol, runtime_checkable

from data_eval.types import Column, ExecutionResult, Schema


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

    def close(self) -> None:
        """Release the underlying connection/resources. Called once at session end."""
        ...


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
