"""Tests for the `FirstDecisive` combinator and the `observed_equivalence` composition.

`FirstDecisive` is exercised with lightweight fake scorers; `observed_equivalence` runs against
an in-process DuckDB engine.
"""

import pytest

from evaldata.platforms.duckdb import DuckDBAdapter
from evaldata.scorers import FirstDecisive, QueryRunner, ScoreContext, Scorer, observed_equivalence
from evaldata.types import (
    DuckDBPlatformRef,
    EvalCase,
    ExecutionResult,
    ExecutionSuccess,
    GoldQuery,
    ResultSetDiff,
    ScoreResult,
    SolverOutput,
    SolverSuccess,
    Sql,
)

_OUTPUT = SolverSuccess(output="SELECT 1")
_RESULT = ExecutionSuccess(rows=[], latency_seconds=0.0)


class _FakeScorer:
    """A `Scorer` returning a canned `ScoreResult`, counting how often it is consulted."""

    def __init__(self, result: ScoreResult) -> None:
        self._result = result
        self.calls = 0

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        self.calls += 1
        return self._result


def _passing(scorer: str) -> ScoreResult:
    return ScoreResult(scorer=scorer, verdict="pass")


def _failing(scorer: str, *, diff: ResultSetDiff | None = None) -> ScoreResult:
    return ScoreResult(scorer=scorer, verdict="fail", diff=diff)


def _inconclusive(scorer: str) -> ScoreResult:
    return ScoreResult(scorer=scorer, verdict="inconclusive")


def _gold_case(gold_sql: str) -> EvalCase:
    return EvalCase(id="c", input="q", expected=GoldQuery(sql=gold_sql), platform=DuckDBPlatformRef(name="x"))


class _NullAdapter:
    """An adapter that is never executed."""

    def execute(self, sql: str) -> ExecutionResult:  # pragma: no cover - never called
        msg = "must not execute SQL"
        raise AssertionError(msg)

    def cancel(self) -> None: ...

    def close(self) -> None: ...


def _null_context(model: str) -> ScoreContext:
    return ScoreContext(queries=QueryRunner(_NullAdapter(), Sql(model), "duckdb", None))


@pytest.mark.unit
class TestFirstDecisive:
    def test_is_a_scorer(self) -> None:
        assert isinstance(FirstDecisive([_FakeScorer(_passing("a"))]), Scorer)

    def test_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one scorer"):
            FirstDecisive([])

    def test_first_passing_member_short_circuits(self) -> None:
        first = _FakeScorer(_passing("first"))
        second = _FakeScorer(_passing("second"))
        case = _gold_case("SELECT 1")
        score = FirstDecisive([first, second]).score(case, _OUTPUT, _RESULT, context=_null_context("SELECT 1"))
        assert score.passed is True
        assert score.scorer == "first"
        assert first.calls == 1
        assert second.calls == 0
        assert score.metadata["first_decisive"] == [{"scorer": "first", "passed": True, "verdict": "pass"}]

    def test_first_failing_member_short_circuits_and_preserves_diff(self) -> None:
        diff = ResultSetDiff(expected_row_count=1, actual_row_count=0, missing_row_count=1)
        first = _FakeScorer(_failing("first", diff=diff))
        second = _FakeScorer(_passing("second"))
        case = _gold_case("SELECT 1")
        score = FirstDecisive([first, second]).score(case, _OUTPUT, _RESULT, context=_null_context("SELECT 1"))
        assert score.passed is False
        assert score.scorer == "first"
        assert score.diff is diff
        assert first.calls == 1
        assert second.calls == 0
        assert score.metadata["first_decisive"] == [{"scorer": "first", "passed": False, "verdict": "fail"}]

    def test_all_inconclusive_returns_last_member(self) -> None:
        first = _FakeScorer(_inconclusive("first"))
        second = _FakeScorer(_inconclusive("second"))
        case = _gold_case("SELECT 1")
        score = FirstDecisive([first, second]).score(case, _OUTPUT, _RESULT, context=_null_context("SELECT 1"))
        assert score.verdict == "inconclusive"
        assert score.scorer == "second"
        assert first.calls == 1
        assert second.calls == 1
        assert score.metadata["first_decisive"] == [
            {"scorer": "first", "passed": False, "verdict": "inconclusive"},
            {"scorer": "second", "passed": False, "verdict": "inconclusive"},
        ]

    def test_inconclusive_then_failing_returns_fail(self) -> None:
        first = _FakeScorer(_inconclusive("first"))
        second = _FakeScorer(_failing("second"))
        case = _gold_case("SELECT 1")
        score = FirstDecisive([first, second]).score(case, _OUTPUT, _RESULT, context=_null_context("SELECT 1"))
        assert score.passed is False
        assert score.scorer == "second"
        assert score.metadata["first_decisive"] == [
            {"scorer": "first", "passed": False, "verdict": "inconclusive"},
            {"scorer": "second", "passed": False, "verdict": "fail"},
        ]

    def test_failing_member_is_not_overridden_by_a_later_passing_member(self) -> None:
        first = _FakeScorer(_failing("first"))
        second = _FakeScorer(_passing("second"))
        case = _gold_case("SELECT 1")
        score = FirstDecisive([first, second]).score(case, _OUTPUT, _RESULT, context=_null_context("SELECT 1"))
        assert score.verdict == "fail"
        assert score.scorer == "first"
        assert first.calls == 1
        assert second.calls == 0

    def test_merges_into_existing_metadata(self) -> None:
        result = ScoreResult(scorer="only", verdict="pass", metadata={"verdicts": []})
        member = _FakeScorer(result)
        case = _gold_case("SELECT 1")
        score = FirstDecisive([member]).score(case, _OUTPUT, _RESULT, context=_null_context("SELECT 1"))
        assert score.metadata["verdicts"] == []
        assert score.metadata["first_decisive"] == [{"scorer": "only", "passed": True, "verdict": "pass"}]


def _duckdb_context(model: str) -> ScoreContext:
    return ScoreContext(queries=QueryRunner(DuckDBAdapter(), Sql(model), "duckdb", None))


def _trail(score: ScoreResult) -> list[str]:
    """The member scorer names recorded in the `first_decisive` trail, in run order."""
    return [entry["scorer"] for entry in score.metadata["first_decisive"]]


@pytest.mark.unit
class TestObservedEquivalence:
    def test_ast_confirms_and_skips_execution(self) -> None:
        model = "select NAME from t where id > 1 and country = 'US'"
        case = _gold_case("SELECT name FROM t WHERE country = 'US' AND id > 1")
        result = ExecutionSuccess(rows=[], latency_seconds=0.0)
        score = observed_equivalence().score(case, _OUTPUT, result, context=_duckdb_context(model))
        assert score.passed is True
        assert _trail(score) == ["semantic_equivalence"]

    def test_ast_inconclusive_then_execution_confirms(self) -> None:
        model = "SELECT 2 AS n UNION ALL SELECT 1 AS n"
        case = _gold_case("SELECT 1 AS n UNION ALL SELECT 2 AS n")
        result = ExecutionSuccess(rows=[{"n": 2}, {"n": 1}], latency_seconds=0.0)
        score = observed_equivalence().score(case, _OUTPUT, result, context=_duckdb_context(model))
        assert score.passed is True
        assert _trail(score) == ["semantic_equivalence", "result_set_equivalence"]

    def test_execution_refutes_genuinely_different_queries(self) -> None:
        model = "SELECT 1 AS n"
        case = _gold_case("SELECT 2 AS n")
        result = ExecutionSuccess(rows=[{"n": 1}], latency_seconds=0.0)
        score = observed_equivalence().score(case, _OUTPUT, result, context=_duckdb_context(model))
        assert score.passed is False
        assert _trail(score) == ["semantic_equivalence", "result_set_equivalence"]
        assert score.diff is not None
