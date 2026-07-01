"""The dbt Semantic Layer evaluation vertical: query, case, output, and pluggable contracts."""

from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaldata.types import PlatformRef, ScoreResult, SolverError


class MetricQuery(BaseModel):
    """A dbt Semantic Layer query: the metrics to compute and how to slice, filter, and limit them.

    `group_by` holds MetricFlow group-by items (a dimension, an entity, or a time dimension with a
    grain, e.g. `metric_time__month` or `customer__country`). `where` holds MetricFlow filter
    expressions (e.g. `{{ Dimension('order_id__is_food_order') }} = true`). `order_by` holds
    group-by or metric names, each optionally prefixed with `-` for descending.
    """

    model_config = ConfigDict(extra="forbid")

    metrics: Annotated[list[str], Field(min_length=1)]
    group_by: list[str] = Field(default_factory=list)
    where: list[str] = Field(default_factory=list)
    order_by: list[str] = Field(default_factory=list)
    limit: Annotated[int, Field(ge=0)] | None = None


class MetricCase(BaseModel):
    """One Semantic Layer evaluation case: a question, a gold metric query, and its resolution context."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    input: Annotated[str, Field(min_length=1)]
    gold: MetricQuery
    platform: PlatformRef
    target_dir: Annotated[str, Field(min_length=1)]
    profiles_dir: str | None = None
    sl_context: str = ""
    metadata: dict[str, object] = Field(default_factory=dict)


class MetricSolverOutput(BaseModel):
    """A Semantic Layer solver's output: either a candidate `query` or an `error` (exactly one set)."""

    model_config = ConfigDict(extra="forbid")

    query: MetricQuery | None = None
    error: SolverError | None = None
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    latency_seconds: Annotated[float, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_of_query_or_error(self) -> "MetricSolverOutput":
        """Enforce that exactly one of `query`/`error` is set.

        Returns:
            The validated `MetricSolverOutput`.

        Raises:
            ValueError: If both or neither of `query` and `error` are set.
        """
        if (self.query is None) == (self.error is None):
            msg = "MetricSolverOutput requires exactly one of 'query' or 'error' to be set"
            raise ValueError(msg)
        return self


@runtime_checkable
class MetricSolver(Protocol):
    """Produces a `MetricSolverOutput` for a `MetricCase`."""

    def solve(self, case: MetricCase) -> MetricSolverOutput:
        """Produce a candidate metric query for `case`."""
        ...


@runtime_checkable
class MetricScorer(Protocol):
    """Scores a candidate metric query against a case's gold query."""

    def score(self, case: MetricCase, query: MetricQuery) -> ScoreResult:
        """Decide pass/fail with diagnostics for `case` given the candidate `query`."""
        ...
