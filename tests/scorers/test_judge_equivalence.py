"""The judge equivalence preset: `judged_equivalence` cascades AST then an LLM judge.

AST confirms equivalence without running a query; when it cannot, the judge decides. The
composition executes no SQL, so the adapter must never be called.
"""

import pytest

from evaldata.llm import StubLlm
from evaldata.scorers import (
    FirstDecisive,
    LlmJudge,
    QueryRunner,
    ScoreContext,
    judged_equivalence,
    sql_equivalence_judge,
)
from evaldata.scorers.llm_judge import JudgeReply
from evaldata.scorers.semantic_equivalence import SemanticEquivalence
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    DuckDBPlatformRef,
    EvalCase,
    ExecutionResult,
    ExecutionSuccess,
    GoldQuery,
    SolverSuccess,
    Sql,
)

_OUTPUT = SolverSuccess(output="SELECT 1")
_RESULT = ExecutionSuccess(rows=[], latency_seconds=0.0)


class _NullAdapter:
    """An adapter that is never executed — the judge composition touches no warehouse."""

    def execute(self, sql: str) -> ExecutionResult:  # pragma: no cover - never called
        msg = "must not execute SQL"
        raise AssertionError(msg)

    def cancel(self) -> None: ...

    def close(self) -> None: ...


def _context(model: str, dialect: Dialect = "duckdb") -> ScoreContext:
    return ScoreContext(queries=QueryRunner(_NullAdapter(), Sql(model), dialect, None))


def _gold_case(gold_sql: str) -> EvalCase:
    return EvalCase(id="c", input="q", expected=GoldQuery(sql=gold_sql), platform=DuckDBPlatformRef(name="x"))


def _trail(score) -> list[str]:
    return [entry["scorer"] for entry in score.metadata["first_decisive"]]


@pytest.mark.unit
class TestSqlEquivalenceJudge:
    def test_returns_a_configured_llm_judge(self) -> None:
        judge = sql_equivalence_judge("openai/gpt-4o-mini")
        assert isinstance(judge, LlmJudge)
        assert "same rows on every database" in judge._criteria
        examples = judge._examples
        assert any(e.actual_output == "SELECT SUM(quantity) FROM orders" and e.score == 0.0 for e in examples)


@pytest.mark.unit
class TestJudgedEquivalence:
    def test_composes_semantic_then_judge(self) -> None:
        composition = judged_equivalence("openai/gpt-4o-mini")
        assert isinstance(composition, FirstDecisive)
        members = composition._scorers
        assert isinstance(members[0], SemanticEquivalence)
        assert isinstance(members[1], LlmJudge)

    def test_ast_confirms_and_judge_not_consulted(self) -> None:
        composition = judged_equivalence(StubLlm(JudgeReply(score=0.0, reason="never asked")))
        case = _gold_case("SELECT name FROM t WHERE country = 'US' AND id > 1")
        model = "select NAME from t where id > 1 and country = 'US'"
        score = composition.score(case, _OUTPUT, _RESULT, context=_context(model))
        assert score.passed is True
        assert score.basis == "proven"
        assert _trail(score) == ["semantic_equivalence"]
        judge = composition._scorers[1]
        assert judge._llm.prompts == []

    def test_ast_inconclusive_then_judge_passes(self) -> None:
        composition = judged_equivalence(StubLlm(JudgeReply(score=0.9, reason="equivalent enough")))
        case = _gold_case("SELECT 2 AS n")
        score = composition.score(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))
        assert score.verdict == "pass"
        assert score.basis == "judged"
        assert _trail(score) == ["semantic_equivalence", "llm_judge"]

    def test_ast_inconclusive_then_judge_fails(self) -> None:
        composition = judged_equivalence(StubLlm(JudgeReply(score=0.1, reason="different")))
        case = _gold_case("SELECT 2 AS n")
        score = composition.score(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))
        assert score.verdict == "fail"
        assert _trail(score) == ["semantic_equivalence", "llm_judge"]
