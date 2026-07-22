"""The dbt Semantic Layer evaluation vertical: query, case, output, and pluggable contracts."""

from typing import Annotated, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

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


class _MetricSolverOutputBase(BaseModel):
    """Usage data shared by Semantic Layer solver outcomes."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    latency_seconds: Annotated[float, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class MetricSolverSuccess(_MetricSolverOutputBase):
    """A metric query produced successfully by a Semantic Layer solver."""

    status: Literal["success"] = "success"
    query: MetricQuery


class MetricSolverFailure(_MetricSolverOutputBase):
    """A Semantic Layer solver failure."""

    status: Literal["failure"] = "failure"
    error: SolverError


MetricSolverOutput: TypeAlias = Annotated[
    MetricSolverSuccess | MetricSolverFailure,
    Field(discriminator="status"),
]


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
