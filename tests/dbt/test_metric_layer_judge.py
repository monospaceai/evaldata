"""Tests for `MetricLayerJudge` and the `metric_layer_equivalence` cascade, driven by a `StubLlm`."""

import pytest

from evaldata.dbt import (
    MetricCase,
    MetricFirstDecisive,
    MetricLayerJudge,
    MetricQuery,
    metric_layer_equivalence,
    strict_metric_equivalence,
)
from evaldata.dbt.metric_layer_judge import _render_query
from evaldata.llm import StubLlm
from evaldata.scorers.llm_judge import JudgeReply
from evaldata.types import LlmError, PlatformRef

pytestmark = pytest.mark.unit

PLATFORM = PlatformRef(name="duck", kind="duckdb")


def _case() -> MetricCase:
    return MetricCase(
        id="c",
        input="Total revenue by month?",
        gold=MetricQuery(metrics=["revenue"]),
        platform=PLATFORM,
        target_dir="t",
    )


def _judge(reply: object) -> MetricLayerJudge:
    return MetricLayerJudge(model=StubLlm(reply))


def test_render_query_includes_all_parts() -> None:
    text = _render_query(
        MetricQuery(
            metrics=["revenue"], group_by=["metric_time__month"], where=["a = 1"], order_by=["-revenue"], limit=5
        )
    )
    assert "metrics: revenue" in text
    assert "group by: metric_time__month" in text
    assert "where: a = 1" in text
    assert "order by: -revenue" in text
    assert "limit: 5" in text


def test_render_query_omits_absent_parts() -> None:
    assert _render_query(MetricQuery(metrics=["revenue"])) == "metrics: revenue"


def test_pass_above_threshold() -> None:
    score = _judge(JudgeReply(reason="same metrics and grain", score=0.9)).score(
        _case(), MetricQuery(metrics=["revenue"])
    )
    assert score.verdict == "pass"
    assert score.score == 0.9
    assert score.basis == "judged"
    assert score.explanation == "same metrics and grain"


def test_fail_below_threshold() -> None:
    score = _judge(JudgeReply(reason="different metric", score=0.1)).score(_case(), MetricQuery(metrics=["orders"]))
    assert score.verdict == "fail"
    assert score.score == 0.1


def test_score_is_clamped() -> None:
    high = _judge(JudgeReply(reason="x", score=1.5)).score(_case(), MetricQuery(metrics=["revenue"]))
    low = _judge(JudgeReply(reason="x", score=-0.5)).score(_case(), MetricQuery(metrics=["revenue"]))
    assert high.score == 1.0
    assert high.verdict == "pass"
    assert low.score == 0.0
    assert low.verdict == "fail"


def test_inconclusive_on_grader_error() -> None:
    score = _judge(LlmError(kind="rate_limit", message="429")).score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "inconclusive"
    assert score.metadata["error"]["kind"] == "rate_limit"


def test_prompt_includes_question_and_both_queries() -> None:
    stub = StubLlm(JudgeReply(reason="x", score=1.0))
    MetricLayerJudge(model=stub).score(_case(), MetricQuery(metrics=["revenue"], group_by=["metric_time__day"]))
    prompt = stub.prompts[-1]
    assert "Total revenue by month?" in prompt
    assert "Candidate query:" in prompt
    assert "metric_time__day" in prompt
    assert "Reference query:" in prompt


def test_custom_threshold_and_criteria() -> None:
    judge = MetricLayerJudge(model=StubLlm(JudgeReply(reason="x", score=0.6)), criteria="be strict", threshold=0.8)
    assert judge.score(_case(), MetricQuery(metrics=["revenue"])).verdict == "fail"


def test_model_string_is_recorded() -> None:
    judge = MetricLayerJudge("openai/gpt-4o-mini")
    assert judge._model == "openai/gpt-4o-mini"


def test_metric_layer_equivalence_builds_full_cascade() -> None:
    cascade = metric_layer_equivalence(StubLlm(JudgeReply(reason="x", score=1.0)))
    assert isinstance(cascade, MetricFirstDecisive)
    assert len(cascade._scorers) == 3


def test_strict_metric_equivalence_has_no_judge() -> None:
    cascade = strict_metric_equivalence()
    assert isinstance(cascade, MetricFirstDecisive)
    assert len(cascade._scorers) == 2
