"""Tests for `LlmJudge` — the LLM-as-judge scorer, driven through a `StubLlm`.

The judge talks only to the `Llm` seam, so its unit tests inject a `StubLlm` (no litellm, no
network). The litellm backend itself is covered by its own tests.
"""

import os

import pytest
from pydantic import ValidationError

from evaldata.llm import StubLlm
from evaldata.scorers import QueryRunner, ScoreContext, Scorer
from evaldata.scorers.llm_judge import (
    JUDGE_INSTRUCTION,
    SCORER_NAME,
    JudgeExample,
    JudgeReply,
    LlmJudge,
    RubricBand,
)
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    DuckDBPlatformRef,
    EvalCase,
    ExecutionResult,
    ExecutionSuccess,
    Expected,
    GoldQuery,
    LlmError,
    SolverSuccess,
    Sql,
    UntypedResultSet,
)

_OUTPUT = SolverSuccess(output="SELECT 1")
_RESULT = ExecutionSuccess(rows=[], latency_seconds=0.0)


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
        platform=DuckDBPlatformRef(name="x"),
    )


@pytest.mark.unit
class TestLlmJudge:
    def test_score_at_or_above_threshold_passes(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="great"))
        result = LlmJudge(model=stub, criteria="is it correct?").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "pass"
        assert result.score == pytest.approx(0.9)
        assert result.basis == "judged"
        assert result.explanation == "great"

    def test_score_below_threshold_fails(self) -> None:
        stub = StubLlm(JudgeReply(score=0.2, reason="off"))
        result = LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "fail"
        assert result.score == pytest.approx(0.2)
        assert result.basis == "judged"

    def test_malformed_output_is_inconclusive(self) -> None:
        err = LlmError(kind="malformed_output", message="grader returned malformed output")
        result = LlmJudge(model=StubLlm(err), criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "inconclusive"
        assert result.score is None
        assert result.basis is None
        assert "grader call failed" in (result.explanation or "")
        assert result.metadata["error"]["kind"] == "malformed_output"

    def test_provider_error_is_inconclusive_with_metadata(self) -> None:
        err = LlmError(kind="api_error", message="boom", provider="openai")
        result = LlmJudge(model=StubLlm(err), criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "inconclusive"
        assert result.score is None
        assert "grader call failed" in (result.explanation or "")
        assert result.metadata["error"]["kind"] == "api_error"
        assert "boom" in result.metadata["error"]["message"]

    def test_score_above_one_clamped(self) -> None:
        stub = StubLlm(JudgeReply(score=1.7, reason="r"))
        result = LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "pass"
        assert result.score == pytest.approx(1.0)

    def test_score_below_zero_clamped(self) -> None:
        stub = StubLlm(JudgeReply(score=-0.2, reason="r"))
        result = LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "fail"
        assert result.score == pytest.approx(0.0)

    def test_empty_reason_is_no_explanation(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason=""))
        result = LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "pass"
        assert result.explanation is None

    def test_metadata_carries_source_and_grader_model(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        result = LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.metadata["source"] == "llm_judge"
        assert result.metadata["grader_model"] == "StubLlm"

    def test_grader_model_string_recorded(self) -> None:
        result = LlmJudge(model="my-grader", criteria="c")
        assert result._model == "my-grader"

    def test_threshold_boundary_passes(self) -> None:
        stub = StubLlm(JudgeReply(score=0.5, reason="r"))
        result = LlmJudge(model=stub, criteria="c", threshold=0.5).score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.verdict == "pass"

    def test_prompt_carries_criteria_question_and_model_sql(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="the answer must be exact").score(
            _case(), _OUTPUT, _RESULT, context=_context(model="SELECT count(*) FROM tracks")
        )
        prompt = stub.prompts[-1]
        assert "the answer must be exact" in prompt
        assert "How many tracks?" in prompt
        assert "SELECT count(*) FROM tracks" in prompt

    def test_gold_query_included_only_for_gold_query_expected(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c").score(
            _case(GoldQuery(sql="SELECT 42 AS gold")), _OUTPUT, _RESULT, context=_context()
        )
        assert "SELECT 42 AS gold" in stub.prompts[-1]

    def test_gold_query_absent_for_non_gold_expected(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c").score(
            _case(UntypedResultSet(rows=[{"n": 1}])), _OUTPUT, _RESULT, context=_context()
        )
        assert "Expected Output" not in stub.prompts[-1]

    def test_show_limits_fields(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c", show=["input"]).score(
            _case(), _OUTPUT, _RESULT, context=_context(model="SELECT secret")
        )
        prompt = stub.prompts[-1]
        assert "How many tracks?" in prompt
        assert "SELECT secret" not in prompt

    def test_show_can_omit_input(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c", show=["actual_output"]).score(
            _case(), _OUTPUT, _RESULT, context=_context(model="SELECT secret")
        )
        prompt = stub.prompts[-1]
        assert "How many tracks?" not in prompt
        assert "SELECT secret" in prompt

    def test_default_instruction_and_output_format_appear(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        prompt = stub.prompts[-1]
        assert JUDGE_INSTRUCTION in prompt
        assert "Return a JSON object with a numeric `score` in [0.0, 1.0]" in prompt

    def test_steps_render_as_numbered_block(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c", steps=["read the question", "check the SQL"]).score(
            _case(), _OUTPUT, _RESULT, context=_context()
        )
        prompt = stub.prompts[-1]
        assert "Evaluation steps:" in prompt
        assert "1. read the question" in prompt
        assert "2. check the SQL" in prompt

    def test_steps_absent_when_not_provided(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert "Evaluation steps:" not in stub.prompts[-1]

    def test_rubric_renders_when_provided(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        rubric = [
            RubricBand(min_score=0.0, max_score=0.3, description="wrong result"),
            RubricBand(min_score=0.7, max_score=1.0, description="correct result"),
        ]
        LlmJudge(model=stub, criteria="c", rubric=rubric).score(_case(), _OUTPUT, _RESULT, context=_context())
        prompt = stub.prompts[-1]
        assert "Scoring rubric:" in prompt
        assert "- 0.0-0.3: wrong result" in prompt
        assert "- 0.7-1.0: correct result" in prompt

    def test_rubric_absent_when_not_provided(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert "Scoring rubric:" not in stub.prompts[-1]

    def test_rubric_band_rejects_inverted_bounds(self) -> None:
        with pytest.raises(ValidationError):
            RubricBand(min_score=0.8, max_score=0.2, description="x")

    def test_examples_render_only_present_fields(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        examples = [
            JudgeExample(actual_output="SELECT 1", score=0.2, reason="too few rows"),
            JudgeExample(
                actual_output="SELECT count(*) FROM t",
                score=0.9,
                reason="matches gold",
                input="how many?",
                expected_output="SELECT count(*) FROM t",
            ),
        ]
        LlmJudge(model=stub, criteria="c", examples=examples).score(_case(), _OUTPUT, _RESULT, context=_context())
        prompt = stub.prompts[-1]
        assert "Examples:" in prompt
        # The first example carries no input; the block runs up to its reason line.
        first_block = prompt.split("Reason: too few rows")[0].split("Examples:")[1]
        assert "Input:" not in first_block
        assert "Actual Output:\nSELECT 1" in first_block
        assert "Score: 0.2" in first_block
        # The second example carries both an input and an expected output.
        assert "Input:\nhow many?" in prompt
        assert "Reason: matches gold" in prompt

    def test_examples_absent_when_not_provided(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert "Examples:" not in stub.prompts[-1]

    def test_judge_example_rejects_out_of_range_score(self) -> None:
        with pytest.raises(ValidationError):
            JudgeExample(actual_output="SELECT 1", score=1.5, reason="r")

    def test_scorer_name(self) -> None:
        stub = StubLlm(JudgeReply(score=0.9, reason="r"))
        result = LlmJudge(model=stub, criteria="c").score(_case(), _OUTPUT, _RESULT, context=_context())
        assert result.scorer == SCORER_NAME

    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(LlmJudge(model=StubLlm(JudgeReply(score=0.9, reason="r")), criteria="c"), Scorer)


@pytest.mark.e2e
@pytest.mark.manual
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
