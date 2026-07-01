"""Preset Semantic Layer scorers: the ready-made `metric_layer_equivalence` cascade."""

from evaldata.dbt.combinators import MetricFirstDecisive
from evaldata.dbt.metric_layer_judge import MetricLayerJudge
from evaldata.dbt.metric_result_equivalence import MetricResultEquivalence
from evaldata.dbt.metric_spec_equivalence import MetricSpecEquivalence
from evaldata.llm import Llm


def metric_layer_equivalence(model: str | Llm) -> MetricFirstDecisive:
    """Return a cost-ordered `MetricFirstDecisive` cascade: spec-compare → run-compare → LLM judge.

    Args:
        model: A litellm grader-model identifier, or an `Llm` to use directly, for the judge tier.

    Returns:
        A `MetricFirstDecisive` over `MetricSpecEquivalence`, `MetricResultEquivalence`, and
        `MetricLayerJudge(model)`.
    """
    return MetricFirstDecisive([MetricSpecEquivalence(), MetricResultEquivalence(), MetricLayerJudge(model)])
