"""`QueryRunner`: a budget-aware handle scorers use to run derived SQL in-platform."""

from dataclasses import dataclass
from typing import Any

from dataeval.platforms.base import PlatformAdapter, execute_within_budget
from dataeval.scorers.sql import Dialect
from dataeval.types import ExecutionResult, Sql


@dataclass(frozen=True)
class ScalarResult:
    """The single cell returned by a derived query, or an error (errors-as-values).

    Attributes:
        value: The single cell value, or `None` when `error` is set.
        error: A failure message, or `None` on success.
        latency_seconds: Wall-clock time the underlying query took.
    """

    value: Any | None
    error: str | None
    latency_seconds: float


class QueryRunner:
    """Runs derived SQL against a case's platform, drawing on a shared cost budget.

    Holds a live adapter, the model's SQL, the case dialect, and a remaining-time pool
    seeded from the case budget. Each completed query decrements the pool by its
    `latency_seconds`; once the pool is exhausted, further runs short-circuit to an
    errors-as-value `ExecutionResult` without touching the adapter. A `None` budget means
    the pool is unbounded.
    """

    def __init__(self, adapter: PlatformAdapter, model_sql: Sql, dialect: Dialect, budget: float | None) -> None:
        """Bind the runner to a platform and seed its budget pool.

        Args:
            adapter: The platform adapter derived queries execute against.
            model_sql: The model's SQL.
            dialect: The SQLGlot dialect derived queries are built and rendered in.
            budget: The shared remaining-time pool in seconds, or `None` for unbounded.
        """
        self._adapter = adapter
        self._model_sql = model_sql
        self._dialect = dialect
        self._remaining = budget

    @property
    def model_sql(self) -> Sql:
        """The model's SQL."""
        return self._model_sql

    @property
    def dialect(self) -> Dialect:
        """The dialect derived queries are built and rendered in."""
        return self._dialect

    def run(self, sql: Sql) -> ExecutionResult:
        """Run `sql` within the remaining budget, decrementing the pool by its latency.

        Args:
            sql: The SQL statement to execute.

        Returns:
            The adapter's `ExecutionResult`, or an `ExecutionResult` with `error` set when
            the budget pool is already exhausted (the adapter is not invoked in that case).
        """
        if self._remaining is not None and self._remaining <= 0:
            return ExecutionResult(
                rows=[],
                schema=None,
                latency_seconds=0.0,
                error="exceeded cost budget: derived-query budget pool exhausted",
            )
        result = execute_within_budget(self._adapter, sql, self._remaining)
        if self._remaining is not None:
            self._remaining -= result.latency_seconds
        return result

    def scalar(self, sql: Sql) -> ScalarResult:
        """Run `sql` and read back its single cell, or an error (errors-as-values).

        Delegates to `run`, so the budget pool is drawn exactly as for any derived query.
        An underlying `error` is propagated; a result that is not exactly one row by one
        column is itself an error.

        Args:
            sql: The SQL statement to execute; expected to return one row and one column.

        Returns:
            A `ScalarResult` carrying the single cell on success, else `error`.
        """
        result = self.run(sql)
        if result.error is not None:
            return ScalarResult(value=None, error=result.error, latency_seconds=result.latency_seconds)
        if len(result.rows) != 1 or len(result.rows[0]) != 1:
            return ScalarResult(
                value=None,
                error="expected one row and one column",
                latency_seconds=result.latency_seconds,
            )
        (value,) = result.rows[0].values()
        return ScalarResult(value=value, error=None, latency_seconds=result.latency_seconds)
