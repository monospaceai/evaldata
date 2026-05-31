"""`PlatformAdapter` Protocol: the contract every platform integration implements."""

from typing import Protocol, runtime_checkable

from data_eval.types import ExecutionResult


@runtime_checkable
class PlatformAdapter(Protocol):
    """Executes SQL against a data platform; returns rows + schema + latency.

    Required behavior:
        * On success: return `ExecutionResult` with `rows` populated, `schema_`
          populated (each `Column.type` is the driver's NATIVE SQL type string —
          e.g. `"BIGINT"`, `"STRUCT(a INTEGER, b VARCHAR)"`, `"INTEGER[]"`),
          non-negative `latency_seconds`, and `error is None`.
        * On query failure: return `ExecutionResult` with `rows=[]`,
          `schema_=None`, a non-empty `error` string describing the failure,
          and non-negative `latency_seconds`. **Do NOT raise.** (Errors-as-values.)
        * `latency_seconds` measures wall-clock time spent inside `execute()`.

    Semantic comparison of the reported native type strings is the equivalence
    engine's job (single-dialect SQLGlot `DataType.is_type`); adapters never
    map types to a normalized vocabulary.
    """

    def execute(self, sql: str) -> ExecutionResult:
        """Execute one SQL statement and return its structured result."""
        ...

    def close(self) -> None:
        """Release the underlying connection/resources. Called once at session end."""
        ...
