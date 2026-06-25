"""Tests for `LlmJudge` — the litellm-backed LLM-as-judge scorer.

litellm is mocked at its boundary (`completion` + `supports_response_schema`), mirroring the
solver tests; no network is touched.
"""

import os
import types

import litellm
import pytest

from evaldata.scorers import QueryRunner, ScoreContext, Scorer
from evaldata.scorers.llm_judge import SCORER_NAME, JudgeReply, LlmJudge
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    EvalCase,
    ExecutionResult,
    Expected,
    GoldQuery,
    PlatformRef,
    SolverOutput,
    Sql,
    UntypedResultSet,
)

_OUTPUT = SolverOutput(output="SELECT 1")
_RESULT = ExecutionResult(rows=[], latency_seconds=0.0)


class _NullAdapter:
    """An adapter that is never executed — the judge compares text, touching no warehouse."""

    def execute(self, sql: str) -> ExecutionResult:  # pragma: no cover - never called
        msg = "LlmJudge must not execute SQL"
        raise AssertionError(msg)

    def cancel(self) -> None: ...

    def close(self) -> None: ...


def _context(model: str = "SELECT 1 AS n", dialect: Dialect = "duckdb") -> ScoreContext:
    return ScoreContext(queries=QueryRunner(_NullAdapter(), Sql(model), dialect, None))


def _case(expected: Expected | None = None) -> EvalCase:
    return EvalCase(
        id="c",
        input="How many tracks?",
        expected=expected if expected is not None else UntypedResultSet(rows=[]),
        platform=PlatformRef(name="x", kind="duckdb"),
    )


def _stub_response(content: str | None):
    """Build a SimpleNamespace exposing exactly what the judge reads off a response."""
    message = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _reply(score: float, reason: str = "looks correct") -> str:
    return JudgeReply(score=score, reason=reason).model_dump_json()


def _patch(monkeypatch: pytest.MonkeyPatch, response, captured: dict | None = None, *, structured: bool = True) -> None:
    def fake(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return response

    monkeypatch.setattr("litellm.completion", fake)
    monkeypatch.setattr("litellm.supports_response_schema", lambda **kwargs: structured)


@pytest.mark.unit
class TestLlmJudge:
    def test_score_at_or_above_threshold_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(0.9, "great")))
        result = LlmJudge(model="grader", criteria="is it correct?").score(
            _case(), _OUTPUT, _RESULT, context=_context()
        )
        assert result.verdict == "pass"
        assert result.score == pytest.approx(0.9)
        assert result.explanation == "great"

    def test_score_below_threshold_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(0.2)))
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "fail"
        assert result.score == pytest.approx(0.2)

    def test_malformed_output_is_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response("not json"))
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "inconclusive"
        assert result.score is None
        assert result.explanation == "grader returned malformed output"

    def test_none_content_is_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(None))
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "inconclusive"
        assert result.score is None

    def test_api_error_is_inconclusive_with_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):
            raise litellm.APIError(status_code=500, message="boom", llm_provider="openai", model="grader")

        monkeypatch.setattr("litellm.completion", fake)
        monkeypatch.setattr("litellm.supports_response_schema", lambda **kwargs: True)
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "inconclusive"
        assert result.score is None
        assert "grader call failed" in (result.explanation or "")
        assert result.metadata["error"]["kind"] == "api_error"
        assert "boom" in result.metadata["error"]["message"]

    def test_unsupported_structured_output_is_inconclusive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(0.9)), structured=False)
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "inconclusive"
        assert result.score is None
        assert "does not support structured output" in (result.explanation or "")

    def test_score_above_one_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(1.7)))
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "pass"
        assert result.score == pytest.approx(1.0)

    def test_score_below_zero_clamped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(-0.2)))
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "fail"
        assert result.score == pytest.approx(0.0)

    def test_empty_reason_is_no_explanation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(0.9, "")))
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "pass"
        assert result.explanation is None

    def test_metadata_carries_source_and_grader_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(0.9)))
        result = LlmJudge(model="my-grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.metadata["source"] == "llm_judge"
        assert result.metadata["grader_model"] == "my-grader"

    def test_threshold_boundary_passes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(0.5)))
        result = LlmJudge(model="grader", criteria="c", threshold=0.5).score(
            _case(), _OUTPUT, _RESULT, context=_context()
        )
        assert result.verdict == "pass"

    def test_response_format_is_judge_reply(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch(monkeypatch, _stub_response(_reply(0.9)), captured)
        LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert captured["response_format"] is JudgeReply
        assert captured["temperature"] == 0.0

    def test_prompt_carries_criteria_question_and_model_sql(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch(monkeypatch, _stub_response(_reply(0.9)), captured)
        LlmJudge(model="grader", criteria="the answer must be exact").score(
            _case(), _OUTPUT, _RESULT, context=_context(model="SELECT count(*) FROM tracks")
        )
        prompt = captured["messages"][0]["content"]
        assert "the answer must be exact" in prompt
        assert "How many tracks?" in prompt
        assert "SELECT count(*) FROM tracks" in prompt

    def test_gold_query_included_only_for_gold_query_expected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch(monkeypatch, _stub_response(_reply(0.9)), captured)
        LlmJudge(model="grader", criteria="c").score(
            _case(GoldQuery(sql="SELECT 42 AS gold")), _OUTPUT, _RESULT, context=_context()
        )
        assert "SELECT 42 AS gold" in captured["messages"][0]["content"]

    def test_gold_query_absent_for_non_gold_expected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch(monkeypatch, _stub_response(_reply(0.9)), captured)
        LlmJudge(model="grader", criteria="c").score(
            _case(UntypedResultSet(rows=[{"n": 1}])), _OUTPUT, _RESULT, context=_context()
        )
        assert "Reference SQL" not in captured["messages"][0]["content"]

    def test_show_limits_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch(monkeypatch, _stub_response(_reply(0.9)), captured)
        LlmJudge(model="grader", criteria="c", show=["question"]).score(
            _case(), _OUTPUT, _RESULT, context=_context(model="SELECT secret")
        )
        prompt = captured["messages"][0]["content"]
        assert "How many tracks?" in prompt
        assert "SELECT secret" not in prompt

    def test_scorer_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch(monkeypatch, _stub_response(_reply(0.9)))
        result = LlmJudge(model="grader", criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.scorer == SCORER_NAME

    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(LlmJudge(model="grader", criteria="c"), Scorer)


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="set OPENAI_API_KEY to run live grader e2e",
)
def test_live_llm_judge_smoke() -> None:
    judge = LlmJudge(
        model="openai/gpt-4o-mini",
        criteria="The query must return a single integer column named n equal to 1.",
    )
    context = ScoreContext(queries=QueryRunner(_NullAdapter(), Sql("SELECT 1 AS n"), "duckdb", None))
    result = judge.score(_case(), _OUTPUT, _RESULT, context=context)
    assert result.verdict in {"pass", "fail"}
    assert result.score is not None
