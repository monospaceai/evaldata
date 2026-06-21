"""Tests for `SemanticEquivalence` and its checks.

`AstEquivalence` is SQLGlot-only (no warehouse); `ExecutionEquivalence` and the full ladder
run against an in-process DuckDB engine.
"""

import pytest

from evaldata.platforms.duckdb import DuckDBAdapter
from evaldata.scorers import QueryRunner, ScoreContext, Scorer
from evaldata.scorers.semantic_equivalence import (
    SCORER_NAME,
    AstEquivalence,
    EquivalenceCheck,
    ExecutionEquivalence,
    SemanticEquivalence,
    default_equivalence_checks,
)
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    EvalCase,
    ExecutionResult,
    Expected,
    GoldQuery,
    PlatformRef,
    SemanticVerdict,
    SolverOutput,
    Sql,
    UntypedResultSet,
)

_OUTPUT = SolverOutput(output="SELECT 1")
_RESULT = ExecutionResult(rows=[], latency_seconds=0.0)


class _NullAdapter:
    """An adapter that is never executed — AST equivalence touches no warehouse."""

    def execute(self, sql: str) -> ExecutionResult:  # pragma: no cover - never called
        msg = "AstEquivalence must not execute SQL"
        raise AssertionError(msg)

    def cancel(self) -> None: ...

    def close(self) -> None: ...


class _FixedCheck:
    """An `EquivalenceCheck` returning a fixed verdict, counting how often it is consulted."""

    def __init__(self, equivalence: str, *, name: str = "execution") -> None:
        self.name = name
        self.calls = 0
        self._equivalence = equivalence

    def judge(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> SemanticVerdict:
        self.calls += 1
        return SemanticVerdict(method=self.name, equivalence=self._equivalence)  # type: ignore[arg-type]


def _context(model: str, dialect: Dialect = "duckdb") -> ScoreContext:
    return ScoreContext(queries=QueryRunner(_NullAdapter(), Sql(model), dialect, None))


def _gold_case(gold_sql: str) -> EvalCase:
    return EvalCase(id="c", input="q", expected=GoldQuery(sql=gold_sql), platform=PlatformRef(name="x", kind="duckdb"))


def _other_case(expected: Expected) -> EvalCase:
    return EvalCase(id="c", input="q", expected=expected, platform=PlatformRef(name="x", kind="duckdb"))


def _ast(model: str, gold: str) -> SemanticVerdict:
    return AstEquivalence().judge(_gold_case(gold), _OUTPUT, _RESULT, context=_context(model))


@pytest.mark.unit
class TestAstEquivalence:
    def test_identical_queries_confirm(self) -> None:
        verdict = _ast("SELECT a FROM t", "SELECT a FROM t")
        assert verdict.method == "ast"
        assert verdict.equivalence == "equivalent"

    def test_whitespace_and_casing_confirm(self) -> None:
        verdict = _ast("select   A  from   T", "SELECT a FROM t")
        assert verdict.equivalence == "equivalent"

    def test_boolean_commutativity_confirms(self) -> None:
        verdict = _ast("SELECT 1 FROM t WHERE a = 1 AND b = 2", "SELECT 1 FROM t WHERE b = 2 AND a = 1")
        assert verdict.equivalence == "equivalent"

    def test_predicate_subsumption_confirms(self) -> None:
        verdict = _ast("SELECT 1 FROM t WHERE a > 5 AND a > 3", "SELECT 1 FROM t WHERE a > 5")
        assert verdict.equivalence == "equivalent"

    def test_constant_folding_confirms(self) -> None:
        verdict = _ast("SELECT 1 + 1 AS n", "SELECT 2 AS n")
        assert verdict.equivalence == "equivalent"

    def test_different_queries_are_unknown_never_refuted(self) -> None:
        verdict = _ast("SELECT 1 AS n", "SELECT 2 AS n")
        assert verdict.equivalence == "unknown"

    def test_arithmetic_commutativity_is_a_sound_miss(self) -> None:
        # SQLGlot does not reorder commutative arithmetic over columns; inconclusive, not refuted.
        verdict = _ast("SELECT x + 1 AS n FROM t", "SELECT 1 + x AS n FROM t")
        assert verdict.equivalence == "unknown"

    def test_multiple_statements_are_unknown(self) -> None:
        verdict = _ast("SELECT 1; SELECT 2", "SELECT 1")
        assert verdict.equivalence == "unknown"
        assert verdict.detail is not None
        assert "model query" in verdict.detail

    def test_gold_query_failure_is_unknown(self) -> None:
        verdict = _ast("SELECT 1", "SELECT 1; SELECT 2")
        assert verdict.equivalence == "unknown"
        assert verdict.detail is not None
        assert "gold query" in verdict.detail

    def test_unparseable_query_is_unknown(self) -> None:
        verdict = _ast("SELECT FROM WHERE )(", "SELECT 1")
        assert verdict.equivalence == "unknown"

    def test_non_gold_expected_is_unknown(self) -> None:
        case = _other_case(UntypedResultSet(rows=[{"n": 1}]))
        verdict = AstEquivalence().judge(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))
        assert verdict.equivalence == "unknown"
        assert verdict.detail == "expected is not a gold query"


@pytest.mark.unit
class TestSemanticEquivalence:
    def test_is_a_scorer(self) -> None:
        assert isinstance(SemanticEquivalence(), Scorer)

    def test_default_checks_are_ast_then_execution(self) -> None:
        checks = default_equivalence_checks()
        assert all(isinstance(check, EquivalenceCheck) for check in checks)
        assert [check.name for check in checks] == ["ast", "execution"]

    def test_passes_when_a_check_confirms(self) -> None:
        case = _gold_case("SELECT a FROM t")
        score = SemanticEquivalence().score(case, _OUTPUT, _RESULT, context=_context("SELECT a FROM t"))
        assert score.scorer == SCORER_NAME
        assert score.passed is True

    def test_fails_as_undecided_when_no_check_decides(self) -> None:
        case = _gold_case("SELECT 2 AS n")
        score = SemanticEquivalence([AstEquivalence()]).score(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))
        assert score.passed is False
        assert score.explanation == "no check could decide equivalence"

    def test_non_gold_expected_raises_type_error(self) -> None:
        case = _other_case(UntypedResultSet(rows=[{"n": 1}]))
        with pytest.raises(TypeError, match="requires a GoldQuery"):
            SemanticEquivalence().score(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))

    def test_stops_at_first_decisive_verdict(self) -> None:
        first = _FixedCheck("equivalent", name="ast")
        second = _FixedCheck("not_equivalent", name="execution")
        case = _gold_case("SELECT 1")
        score = SemanticEquivalence([first, second]).score(case, _OUTPUT, _RESULT, context=_context("SELECT 1"))
        assert score.passed is True
        assert first.calls == 1
        assert second.calls == 0
        assert len(score.metadata["verdicts"]) == 1

    def test_falls_through_unknown_to_a_later_check(self) -> None:
        first = _FixedCheck("unknown", name="ast")
        second = _FixedCheck("not_equivalent", name="execution")
        case = _gold_case("SELECT 1")
        score = SemanticEquivalence([first, second]).score(case, _OUTPUT, _RESULT, context=_context("SELECT 1"))
        assert score.passed is False
        assert first.calls == 1
        assert second.calls == 1


def _duckdb_context(model: str) -> ScoreContext:
    return ScoreContext(queries=QueryRunner(DuckDBAdapter(), Sql(model), "duckdb", None))


@pytest.mark.unit
class TestExecutionEquivalence:
    def test_matching_result_sets_confirm(self) -> None:
        case = _gold_case("SELECT 1 AS n")
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        verdict = ExecutionEquivalence().judge(case, _OUTPUT, result, context=_duckdb_context("SELECT 1 AS n"))
        assert verdict.method == "execution"
        assert verdict.equivalence == "equivalent"

    def test_differing_result_sets_refute_with_diff(self) -> None:
        case = _gold_case("SELECT 2 AS n")
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        verdict = ExecutionEquivalence().judge(case, _OUTPUT, result, context=_duckdb_context("SELECT 1 AS n"))
        assert verdict.equivalence == "not_equivalent"
        assert verdict.diff is not None

    def test_failed_gold_query_is_unknown(self) -> None:
        case = _gold_case("SELECT * FROM no_such_table_xyz")
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        verdict = ExecutionEquivalence().judge(case, _OUTPUT, result, context=_duckdb_context("SELECT 1 AS n"))
        assert verdict.equivalence == "unknown"
        assert verdict.detail is not None


@pytest.mark.unit
class TestLadder:
    def test_ast_abstains_then_execution_confirms(self) -> None:
        # x+1 vs 1+x: AST cannot canonicalize commutative arithmetic, so execution decides.
        model = "SELECT x + 1 AS n FROM (SELECT 1 AS x)"
        case = _gold_case("SELECT 1 + x AS n FROM (SELECT 1 AS x)")
        result = ExecutionResult(rows=[{"n": 2}], latency_seconds=0.0)
        output = SolverOutput(output=Sql(model))
        score = SemanticEquivalence().score(case, output, result, context=_duckdb_context(model))
        assert score.passed is True
        verdicts = [(v["method"], v["equivalence"]) for v in score.metadata["verdicts"]]
        assert verdicts == [("ast", "unknown"), ("execution", "equivalent")]

    def test_ast_confirmation_skips_execution(self) -> None:
        model = "SELECT 1 AS n"
        case = _gold_case("SELECT 1 AS n")
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        output = SolverOutput(output=Sql(model))
        score = SemanticEquivalence().score(case, output, result, context=_duckdb_context(model))
        assert score.passed is True
        verdicts = [(v["method"], v["equivalence"]) for v in score.metadata["verdicts"]]
        assert verdicts == [("ast", "equivalent")]

    def test_execution_refutes_genuinely_different_queries(self) -> None:
        model = "SELECT 1 AS n"
        case = _gold_case("SELECT 2 AS n")
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        output = SolverOutput(output=Sql(model))
        score = SemanticEquivalence().score(case, output, result, context=_duckdb_context(model))
        assert score.passed is False
        assert score.diff is not None
        verdicts = [(v["method"], v["equivalence"]) for v in score.metadata["verdicts"]]
        assert verdicts == [("ast", "unknown"), ("execution", "not_equivalent")]
