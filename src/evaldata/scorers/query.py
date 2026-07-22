"""`QueryRunner`: a budget-aware handle scorers use to run derived SQL in-platform."""

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from evaldata.platforms.base import PlatformAdapter, TypeResolvingAdapter, execute_within_budget
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    Column,
    ExecutionError,
    ExecutionFailure,
    ExecutionResult,
    Sql,
    TypedSchema,
)


class _ScalarResultBase(BaseModel):
    """Timing shared by scalar query outcomes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    latency_seconds: float


class ScalarSuccess(_ScalarResultBase):
    """A successful scalar query."""

    status: Literal["success"] = "success"
    value: Any


class ScalarFailure(_ScalarResultBase):
    """A failed scalar query."""

    status: Literal["failure"] = "failure"
    error: ExecutionError


ScalarResult: TypeAlias = Annotated[ScalarSuccess | ScalarFailure, Field(discriminator="status")]


class QueryRunner:
    """Runs derived SQL against a case's platform, drawing on a shared cost budget.

    Holds a live adapter, the model's SQL, the case dialect, and a remaining-time pool
    seeded from the case budget. Each completed query decrements the pool by its
    `latency_seconds`; once the pool is exhausted, further runs short-circuit to an
    `ExecutionFailure` without touching the adapter. A `None` budget means the pool is
    unbounded.
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
            The adapter's execution outcome, or an `ExecutionFailure` when the budget pool
            is already exhausted (the adapter is not invoked in that case).
        """
        if self._remaining is not None and self._remaining <= 0:
            return ExecutionFailure(
                latency_seconds=0.0,
                error=ExecutionError(
                    kind="budget_exceeded", message="exceeded cost budget: derived-query budget pool exhausted"
                ),
            )
        result = execute_within_budget(self._adapter, sql, self._remaining)
        if self._remaining is not None:
            self._remaining -= result.latency_seconds
        return result

    def scalar(self, sql: Sql) -> ScalarResult:
        """Run `sql` and read back its single cell, or an error.

        Delegates to `run`, so the budget pool is drawn exactly as for any derived query.
        An underlying `error` is propagated; a result that is not exactly one row by one
        column is itself an error.

        Args:
            sql: The SQL statement to execute; expected to return one row and one column.

        Returns:
            A `ScalarSuccess` carrying the single cell, or a `ScalarFailure`.
        """
        result = self.run(sql)
        if isinstance(result, ExecutionFailure):
            return ScalarFailure(error=result.error, latency_seconds=result.latency_seconds)
        if len(result.rows) != 1 or len(result.rows[0]) != 1:
            return ScalarFailure(
                error=ExecutionError(kind="query_failed", message="expected one row and one column"),
                latency_seconds=result.latency_seconds,
            )
        (value,) = result.rows[0].values()
        return ScalarSuccess(value=value, latency_seconds=result.latency_seconds)

    def resolved_schema(self, base: TypedSchema, sql: Sql) -> TypedSchema | ExecutionError:
        """Return `base` with its column types resolved to the platform's precise types.

        Backends that already report precise types return `base` unchanged; otherwise the
        adapter's type probe runs through this runner (drawing on the same budget) and its
        types align to `base`'s columns by position, preserving names and nullability.

        Args:
            base: The schema whose column types to resolve, as `execute` reported them.
            sql: The statement that produced `base`, re-probed for precise types.

        Returns:
            `base`, a new `TypedSchema` with precise types, or an `ExecutionError`.
        """
        adapter = self._adapter
        if not isinstance(adapter, TypeResolvingAdapter):
            return base
        probe = self.run(Sql(adapter.type_probe_sql(sql)))
        if isinstance(probe, ExecutionFailure):
            return probe.error
        types = adapter.types_from_probe(probe.rows)
        if isinstance(types, ExecutionError):
            return types
        if len(types) != len(base.root):
            return ExecutionError(
                kind="type_probe_failed",
                message=f"type probe returned {len(types)} column type(s) for a {len(base.root)}-column result",
            )
        return TypedSchema(
            root=[Column(name=c.name, type=t, nullable=c.nullable) for c, t in zip(base.root, types, strict=True)]
        )
