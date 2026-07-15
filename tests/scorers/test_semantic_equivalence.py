"""Tests for `SemanticEquivalence` and its checks.

`AstEquivalence` is SQLGlot-only (no warehouse); `SemanticEquivalence`'s checks compare the
queries, so these tests never execute a query.
"""

import pytest

from evaldata.scorers import QueryRunner, ScoreContext, Scorer
from evaldata.scorers.semantic_equivalence import (
    SCORER_NAME,
    AstEquivalence,
    EquivalenceCheck,
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

    def __init__(self, equivalence: str, *, method: str = "ast") -> None:
        self.method = method
        self.calls = 0
        self._equivalence = equivalence

    def judge(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> SemanticVerdict:
        self.calls += 1
        return SemanticVerdict(method=self.method, equivalence=self._equivalence)  # type: ignore[arg-type]


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

    def test_binary_arithmetic_commutativity_confirms(self) -> None:
        verdict = _ast("SELECT x + 1 AS n FROM t", "SELECT 1 + x AS n FROM t")
        assert verdict.equivalence == "equivalent"

    def test_chained_arithmetic_reassociation_confirms(self) -> None:
        verdict = _ast("SELECT a + b + c AS n FROM t", "SELECT c + b + a AS n FROM t")
        assert verdict.equivalence == "equivalent"

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

    def test_unfoldable_constant_is_unknown(self) -> None:
        # Folding `1.0 / 0` raises during simplification, so the check returns unknown.
        verdict = _ast("SELECT 1.0 / 0 AS n", "SELECT 1 AS n")
        assert verdict.equivalence == "unknown"

    def test_non_gold_expected_is_unknown(self) -> None:
        case = _other_case(UntypedResultSet(rows=[{"n": 1}]))
        verdict = AstEquivalence().judge(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))
        assert verdict.equivalence == "unknown"
        assert verdict.detail == "expected is not a gold query"


def _ast_dialect(model: str, gold: str, dialect: Dialect) -> SemanticVerdict:
    context = ScoreContext(queries=QueryRunner(_NullAdapter(), Sql(model), dialect, None))
    return AstEquivalence().judge(_gold_case(gold), _OUTPUT, _RESULT, context=context)


# (model, gold) pairs that must confirm as "equivalent" via the canonicalization pass.
_CANONICALIZATION_POSITIVES = [
    ("SELECT x + 1 AS n FROM t", "SELECT 1 + x AS n FROM t"),
    ("SELECT x * 2 AS n FROM t", "SELECT 2 * x AS n FROM t"),
    ("SELECT a + b AS n FROM t", "SELECT b + a AS n FROM t"),
    ("SELECT a * b AS n FROM t", "SELECT b * a AS n FROM t"),
    ("SELECT a + b + c AS n FROM t", "SELECT c + b + a AS n FROM t"),
    ("SELECT a * b * c AS n FROM t", "SELECT c * b * a AS n FROM t"),
    ("SELECT 1 FROM t WHERE a AND b AND c", "SELECT 1 FROM t WHERE c AND b AND a"),
    ("SELECT 1 FROM t WHERE a OR b OR c", "SELECT 1 FROM t WHERE c OR a OR b"),
    ("SELECT 1 FROM t WHERE a = 1", "SELECT 1 FROM t WHERE 1 = a"),
    ("SELECT 1 FROM t WHERE a > 5", "SELECT 1 FROM t WHERE 5 < a"),
    ("SELECT 1 + 1 AS n", "SELECT 2 AS n"),
    ("SELECT 1 FROM t WHERE NOT (a AND b)", "SELECT 1 FROM t WHERE NOT a OR NOT b"),
    ("SELECT 1 FROM t WHERE NOT (a > 5)", "SELECT 1 FROM t WHERE a <= 5"),
    ("SELECT 1 FROM t WHERE a > 5 AND a > 3", "SELECT 1 FROM t WHERE a > 5"),
    ("SELECT 1 FROM t WHERE x IN (3, 1, 2)", "SELECT 1 FROM t WHERE x IN (1, 2, 3)"),
    ("SELECT nvl(a, 0) AS n FROM t", "SELECT coalesce(a, 0) AS n FROM t"),
]

# (model, gold) pairs that must NOT confirm as "equivalent".
_CANONICALIZATION_MUST_DIFFER = [
    ("SELECT 1 FROM a JOIN b ON a.id = b.id", "SELECT 1 FROM b JOIN a ON a.id = b.id"),
    ("SELECT a FROM t UNION ALL SELECT b FROM t", "SELECT b FROM t UNION ALL SELECT a FROM t"),
    ("SELECT a, b FROM t", "SELECT b, a FROM t"),
    ("SELECT 1 FROM t GROUP BY a, b", "SELECT 1 FROM t GROUP BY b, a"),
    ("SELECT a FROM t ORDER BY a, b", "SELECT a FROM t ORDER BY b, a"),
    ("SELECT a FROM t ORDER BY a ASC", "SELECT a FROM t ORDER BY a DESC"),
    ("SELECT DISTINCT a FROM t", "SELECT a FROM t"),
    ("SELECT a - b AS n FROM t", "SELECT b - a AS n FROM t"),
    ("SELECT a / b AS n FROM t", "SELECT b / a AS n FROM t"),
    ("SELECT 1 FROM t WHERE NOT (a = b)", "SELECT 1 FROM t WHERE a = b"),
    ("SELECT 1 AS n", "SELECT 2 AS n"),
    ("SELECT 1 FROM t WHERE a <=> b", "SELECT 1 FROM t WHERE a = b"),
]

# `^` parses as bitwise xor under Databricks but as exponentiation under DuckDB, so these
# pairs are checked under Databricks.
_BITWISE_POSITIVES = [
    ("SELECT a & b & c AS n FROM t", "SELECT c & a & b AS n FROM t"),
    ("SELECT a | b | c AS n FROM t", "SELECT c | a | b AS n FROM t"),
    ("SELECT a ^ b ^ c AS n FROM t", "SELECT c ^ a ^ b AS n FROM t"),
]


@pytest.mark.unit
class TestCanonicalization:
    @pytest.mark.parametrize(("model", "gold"), _CANONICALIZATION_POSITIVES)
    def test_canonicalization_confirms(self, model: str, gold: str) -> None:
        assert _ast(model, gold).equivalence == "equivalent"

    @pytest.mark.parametrize(("model", "gold"), _CANONICALIZATION_MUST_DIFFER)
    def test_distinct_queries_are_never_confirmed(self, model: str, gold: str) -> None:
        assert _ast(model, gold).equivalence != "equivalent"

    @pytest.mark.parametrize(("model", "gold"), _BITWISE_POSITIVES)
    def test_bitwise_reassociation_confirms(self, model: str, gold: str) -> None:
        assert _ast_dialect(model, gold, "databricks").equivalence == "equivalent"


# Non-deterministic calls that must return unknown (never confirmed), checked under Databricks since
# `monotonically_increasing_id`/`spark_partition_id`/`input_file_name` are Spark builtins.
_NONDETERMINISTIC_QUERIES = [
    "SELECT rand() AS n FROM t",
    "SELECT monotonically_increasing_id() AS n FROM t",
    "SELECT spark_partition_id() AS n FROM t",
    "SELECT input_file_name() AS n FROM t",
]


@pytest.mark.unit
class TestNonDeterminism:
    def test_identical_nondeterministic_queries_are_not_confirmed(self) -> None:
        verdict = _ast("SELECT rand() AS n FROM t", "SELECT rand() AS n FROM t")
        assert verdict.equivalence != "equivalent"

    @pytest.mark.parametrize("query", _NONDETERMINISTIC_QUERIES)
    def test_named_nondeterministic_builtins_return_unknown(self, query: str) -> None:
        verdict = _ast_dialect(query, query, "databricks")
        assert verdict.equivalence == "unknown"
        assert verdict.detail is not None
        assert "non-deterministic" in verdict.detail

    def test_deterministic_query_still_confirms(self) -> None:
        verdict = _ast("SELECT a FROM t", "SELECT a FROM t")
        assert verdict.equivalence == "equivalent"


@pytest.mark.unit
class TestSemanticEquivalence:
    def test_is_a_scorer(self) -> None:
        assert isinstance(SemanticEquivalence(), Scorer)

    def test_default_checks_are_ast_only(self) -> None:
        checks = default_equivalence_checks()
        assert all(isinstance(check, EquivalenceCheck) for check in checks)
        assert [check.method for check in checks] == ["ast"]

    def test_passes_when_a_check_confirms(self) -> None:
        case = _gold_case("SELECT a FROM t")
        score = SemanticEquivalence().score(case, _OUTPUT, _RESULT, context=_context("SELECT a FROM t"))
        assert score.scorer == SCORER_NAME
        assert score.passed is True

    def test_confirmed_result_is_proven(self) -> None:
        case = _gold_case("SELECT a FROM t")
        score = SemanticEquivalence().score(case, _OUTPUT, _RESULT, context=_context("SELECT a FROM t"))
        assert score.basis == "proven"

    def test_inconclusive_when_no_check_decides(self) -> None:
        case = _gold_case("SELECT 2 AS n")
        score = SemanticEquivalence([AstEquivalence()]).score(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))
        assert score.verdict == "inconclusive"
        assert score.passed is False
        assert score.basis is None
        assert score.explanation == "no semantic check could confirm equivalence"

    def test_non_gold_expected_is_inconclusive(self) -> None:
        case = _other_case(UntypedResultSet(rows=[{"n": 1}]))
        score = SemanticEquivalence().score(case, _OUTPUT, _RESULT, context=_context("SELECT 1 AS n"))
        assert score.verdict == "inconclusive"
        assert score.passed is False
        assert score.metadata.get("scorer_misconfigured") is True
        assert score.explanation is not None
        assert "GoldQuery" in score.explanation
        assert "UntypedResultSet" in score.explanation

    def test_stops_at_first_decisive_verdict(self) -> None:
        first = _FixedCheck("equivalent", method="ast")
        second = _FixedCheck("equivalent", method="ast")
        case = _gold_case("SELECT 1")
        score = SemanticEquivalence([first, second]).score(case, _OUTPUT, _RESULT, context=_context("SELECT 1"))
        assert score.passed is True
        assert first.calls == 1
        assert second.calls == 0
        assert len(score.metadata["verdicts"]) == 1

    def test_falls_through_unknown_to_a_later_check(self) -> None:
        first = _FixedCheck("unknown", method="ast")
        second = _FixedCheck("unknown", method="ast")
        case = _gold_case("SELECT 1")
        score = SemanticEquivalence([first, second]).score(case, _OUTPUT, _RESULT, context=_context("SELECT 1"))
        assert score.verdict == "inconclusive"
        assert score.passed is False
        assert first.calls == 1
        assert second.calls == 1
        assert score.explanation == "no semantic check could confirm equivalence"


class TestSubtractionEquivalence:
    """AST equivalence folds `constant - variable` comparisons correctly and soundly."""

    def test_confirms_equivalent_subtraction(self) -> None:
        case = _gold_case("SELECT a FROM t WHERE a = -1")
        score = SemanticEquivalence().score(case, _OUTPUT, _RESULT, context=_context("SELECT a FROM t WHERE 0 - a = 1"))
        assert score.passed is True

    def test_does_not_over_merge_subtraction(self) -> None:
        case = _gold_case("SELECT a FROM t WHERE a = 1")
        score = SemanticEquivalence().score(case, _OUTPUT, _RESULT, context=_context("SELECT a FROM t WHERE 0 - a = 1"))
        assert score.passed is False
