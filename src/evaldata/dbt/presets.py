"""Preset Semantic Layer scorers: the ready-made `metric_layer_equivalence` cascade."""

from evaldata.dbt.combinators import MetricFirstDecisive
from evaldata.dbt.metric_layer_judge import MetricLayerJudge
from evaldata.dbt.metric_result_equivalence import MetricResultEquivalence
from evaldata.dbt.metric_spec_equivalence import MetricSpecEquivalence
from evaldata.llm import Llm


def metric_layer_equivalence(model: str | Llm, *, temperature: float | None = 0.0) -> MetricFirstDecisive:
    """Return a cost-ordered `MetricFirstDecisive` cascade: spec-compare → run-compare → LLM judge.

    Args:
        model: A litellm grader-model identifier, or an `Llm` to use directly, for the judge tier.
        temperature: Sampling temperature for the judge; some reasoning models accept only `1.0`.

    Returns:
        A `MetricFirstDecisive` over `MetricSpecEquivalence`, `MetricResultEquivalence`, and
        `MetricLayerJudge(model)`.
    """
    judge = MetricLayerJudge(model, temperature=temperature)
    return MetricFirstDecisive([MetricSpecEquivalence(), MetricResultEquivalence(), judge])


def strict_metric_equivalence() -> MetricFirstDecisive:
    """Return a strict spec -> run cascade with no judge, scoring a failed run as incorrect.

    Mirrors an execution-accuracy contract: a candidate passes only when its resolved form or its
    result rows match the gold; a query that fails to run counts against it.

    Returns:
        A `MetricFirstDecisive` over `MetricSpecEquivalence` and `MetricResultEquivalence`.
    """
    return MetricFirstDecisive([MetricSpecEquivalence(), MetricResultEquivalence(on_error="fail")])
