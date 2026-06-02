"""Tests for `ExpectationSuiteScorer` — evaluates an `ExpectationSuite` against a result."""

import pytest

from data_eval.scorers import ExpectationSuiteScorer, Scorer
from data_eval.scorers.expectation_suite import SCORER_NAME
from data_eval.types import (
    Column,
    ColumnPresenceExpectation,
    ColumnTypeExpectation,
    EvalCase,
    ExecutionResult,
    ExpectationOutcome,
    ExpectationSuite,
    Expected,
    ExpectedSQL,
    NotNullExpectation,
    PlatformRef,
    RowCountExpectation,
    ScoreResult,
    SolverOutput,
    SqlType,
    UniqueExpectation,
)

_OUTPUT = SolverOutput(output="SELECT ...")


def _case(expected: Expected) -> EvalCase:
    return EvalCase(
        id="c",
        input="q",
        expected=expected,
        platform=PlatformRef(name="x", kind="duckdb"),
    )


def _suite(*expectations: object) -> EvalCase:
    return _case(ExpectationSuite(expectations=list(expectations)))


def _sole(score: ScoreResult) -> ExpectationOutcome:
    """Return the single `ExpectationOutcome` of a single-expectation suite score."""
    assert len(score.outcomes) == 1
    return score.outcomes[0]


@pytest.mark.unit
class TestRowCount:
    def test_pass(self) -> None:
        case = _suite(RowCountExpectation(exact=2))
        result = ExecutionResult(rows=[{"n": 1}, {"n": 2}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.scorer == SCORER_NAME
        assert score.passed is True
        assert score.explanation is None
        outcome = _sole(score)
        assert outcome == ExpectationOutcome(kind="row_count", passed=True, expected="2", actual="2", detail=None)

    def test_fail(self) -> None:
        case = _suite(RowCountExpectation(exact=5))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "expected 5 rows, got 1" in score.explanation
        outcome = _sole(score)
        assert outcome.kind == "row_count"
        assert outcome.passed is False
        assert outcome.expected == "5"
        assert outcome.actual == "1"
        assert outcome.column is None
        assert outcome.count is None
        assert outcome.detail is not None
        assert "expected 5 rows, got 1" in outcome.detail


@pytest.mark.unit
class TestColumnPresence:
    def test_pass_from_schema(self) -> None:
        case = _suite(ColumnPresenceExpectation(columns=["id", "name"]))
        result = ExecutionResult(
            rows=[],
            schema=[Column(name="id", type="BIGINT"), Column(name="name", type="VARCHAR")],
            latency_seconds=0.0,
        )
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        assert _sole(score) == ExpectationOutcome(kind="column_presence", passed=True, detail=None)

    def test_pass_from_rows_when_no_schema(self) -> None:
        case = _suite(ColumnPresenceExpectation(columns=["id"]))
        result = ExecutionResult(rows=[{"id": 1, "name": "x"}], latency_seconds=0.0)
        assert ExpectationSuiteScorer().score(case, _OUTPUT, result).passed is True

    def test_fail_lists_missing(self) -> None:
        case = _suite(ColumnPresenceExpectation(columns=["id", "missing"]))
        result = ExecutionResult(rows=[{"id": 1}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "missing" in score.explanation
        outcome = _sole(score)
        assert outcome.kind == "column_presence"
        assert outcome.passed is False
        assert outcome.column is None
        assert outcome.detail is not None
        assert "missing" in outcome.detail

    def test_fail_when_no_schema_and_no_rows(self) -> None:
        # A non-row-returning result (no schema, no rows) exposes no columns, so any
        # expected column is reported missing.
        case = _suite(ColumnPresenceExpectation(columns=["id"]))
        result = ExecutionResult(rows=[], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "id" in score.explanation


@pytest.mark.unit
class TestColumnType:
    def test_pass(self) -> None:
        case = _suite(ColumnTypeExpectation(column="n", expected_type="BIGINT"))
        result = ExecutionResult(
            rows=[{"n": 1}],
            schema=[Column(name="n", type=SqlType.parse("BIGINT", "duckdb"))],
            latency_seconds=0.0,
        )
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        assert _sole(score) == ExpectationOutcome(
            kind="column_type", passed=True, column="n", expected="BIGINT", actual="BIGINT", detail=None
        )

    def test_pass_aliased_type(self) -> None:
        # INT8 and BIGINT canonicalise to the same duckdb type.
        case = _suite(ColumnTypeExpectation(column="n", expected_type="INT8"))
        result = ExecutionResult(
            rows=[{"n": 1}],
            schema=[Column(name="n", type=SqlType.parse("BIGINT", "duckdb"))],
            latency_seconds=0.0,
        )
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        # Aliased pass: expected/actual preserve the distinct authored vs observed `raw` spellings.
        outcome = _sole(score)
        assert outcome.passed is True
        assert outcome.expected == "INT8"
        assert outcome.actual == "BIGINT"

    def test_fail_mismatch(self) -> None:
        case = _suite(ColumnTypeExpectation(column="n", expected_type="INTEGER"))
        result = ExecutionResult(
            rows=[{"n": 1}],
            schema=[Column(name="n", type=SqlType.parse("BIGINT", "duckdb"))],
            latency_seconds=0.0,
        )
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "column_type" in score.explanation
        outcome = _sole(score)
        assert outcome.kind == "column_type"
        assert outcome.passed is False
        assert outcome.column == "n"
        assert outcome.expected == "INTEGER"
        assert outcome.actual == "BIGINT"

    def test_fail_absent_column(self) -> None:
        case = _suite(ColumnTypeExpectation(column="missing", expected_type="BIGINT"))
        result = ExecutionResult(
            rows=[{"n": 1}],
            schema=[Column(name="n", type=SqlType.parse("BIGINT", "duckdb"))],
            latency_seconds=0.0,
        )
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "not found" in score.explanation
        outcome = _sole(score)
        assert outcome.passed is False
        assert outcome.column == "missing"
        assert outcome.expected == "BIGINT"
        assert outcome.actual is None

    def test_fail_no_schema(self) -> None:
        case = _suite(ColumnTypeExpectation(column="n", expected_type="BIGINT"))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "no column schema available" in score.explanation
        outcome = _sole(score)
        assert outcome.passed is False
        assert outcome.column == "n"
        assert outcome.expected == "BIGINT"
        assert outcome.actual is None


@pytest.mark.unit
class TestNotNull:
    def test_pass(self) -> None:
        case = _suite(NotNullExpectation(column="email"))
        result = ExecutionResult(rows=[{"email": "a"}, {"email": "b"}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        assert _sole(score) == ExpectationOutcome(kind="not_null", passed=True, column="email", count=0, detail=None)

    def test_pass_zero_rows(self) -> None:
        case = _suite(NotNullExpectation(column="email"))
        result = ExecutionResult(rows=[], schema=[Column(name="email", type="VARCHAR")], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        assert _sole(score).count == 0

    def test_fail_reports_count(self) -> None:
        case = _suite(NotNullExpectation(column="email"))
        result = ExecutionResult(
            rows=[{"email": "a"}, {"email": None}, {"email": None}],
            latency_seconds=0.0,
        )
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "2 NULL value(s)" in score.explanation
        outcome = _sole(score)
        assert outcome.kind == "not_null"
        assert outcome.passed is False
        assert outcome.column == "email"
        assert outcome.count == 2

    def test_fail_absent_column(self) -> None:
        case = _suite(NotNullExpectation(column="email"))
        result = ExecutionResult(rows=[{"id": 1}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "not found" in score.explanation
        outcome = _sole(score)
        assert outcome.passed is False
        assert outcome.column == "email"
        assert outcome.count is None


@pytest.mark.unit
class TestUnique:
    def test_pass(self) -> None:
        case = _suite(UniqueExpectation(column="id"))
        result = ExecutionResult(rows=[{"id": 1}, {"id": 2}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        assert _sole(score) == ExpectationOutcome(kind="unique", passed=True, column="id", count=0, detail=None)

    def test_fail_duplicate(self) -> None:
        case = _suite(UniqueExpectation(column="id"))
        result = ExecutionResult(rows=[{"id": 1}, {"id": 1}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "duplicated value(s)" in score.explanation
        outcome = _sole(score)
        assert outcome.kind == "unique"
        assert outcome.passed is False
        assert outcome.column == "id"
        assert outcome.count == 1

    def test_fail_null_duplicates(self) -> None:
        # dbt semantics: NULLs compare as equal, so two NULLs are a duplicate.
        case = _suite(UniqueExpectation(column="id"))
        result = ExecutionResult(rows=[{"id": None}, {"id": None}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert _sole(score).count == 1

    def test_unhashable_values_duplicate(self) -> None:
        case = _suite(UniqueExpectation(column="c"))
        result = ExecutionResult(rows=[{"c": [1, 2]}, {"c": [1, 2]}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert _sole(score).count == 1

    def test_unhashable_values_distinct(self) -> None:
        case = _suite(UniqueExpectation(column="c"))
        result = ExecutionResult(rows=[{"c": [1]}, {"c": [2]}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        assert _sole(score).count == 0

    def test_fail_absent_column(self) -> None:
        case = _suite(UniqueExpectation(column="id"))
        result = ExecutionResult(rows=[{"x": 1}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.explanation is not None
        assert "not found" in score.explanation
        outcome = _sole(score)
        assert outcome.passed is False
        assert outcome.column == "id"
        assert outcome.count is None


@pytest.mark.unit
class TestSuiteAggregation:
    def test_execution_error_passthrough(self) -> None:
        case = _suite(RowCountExpectation(exact=1))
        result = ExecutionResult(rows=[], latency_seconds=0.0, error="boom")
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.diff is None
        assert score.outcomes == []
        assert score.explanation is not None
        assert "boom" in score.explanation

    def test_raises_on_non_expectation_suite(self) -> None:
        case = _case(ExpectedSQL(sql="SELECT 1"))
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.0)
        with pytest.raises(TypeError, match="ExpectationSuite"):
            ExpectationSuiteScorer().score(case, _OUTPUT, result)

    def test_aggregates_multiple_failures(self) -> None:
        case = _suite(RowCountExpectation(exact=5), NotNullExpectation(column="email"))
        result = ExecutionResult(rows=[{"email": None}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        assert score.diff is None
        assert score.explanation is not None
        assert "2 expectation(s) failed" in score.explanation
        assert "row_count" in score.explanation
        assert "not_null" in score.explanation
        # One outcome per expectation, in suite order, both failing.
        assert [o.kind for o in score.outcomes] == ["row_count", "not_null"]
        assert all(o.passed is False for o in score.outcomes)
        # The prose is derived from the outcomes' detail lines.
        for outcome in score.outcomes:
            assert outcome.detail is not None
            assert outcome.detail in score.explanation

    def test_mixed_suite_exposes_pass_and_fail_outcomes(self) -> None:
        case = _suite(RowCountExpectation(exact=1), NotNullExpectation(column="email"))
        result = ExecutionResult(rows=[{"email": None}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is False
        row_count, not_null = score.outcomes
        assert row_count.kind == "row_count"
        assert row_count.passed is True
        assert row_count.detail is None
        assert not_null.kind == "not_null"
        assert not_null.passed is False
        assert score.explanation is not None
        assert "1 expectation(s) failed" in score.explanation

    def test_all_pass(self) -> None:
        case = _suite(
            RowCountExpectation(exact=1),
            ColumnPresenceExpectation(columns=["id"]),
            UniqueExpectation(column="id"),
            NotNullExpectation(column="id"),
        )
        result = ExecutionResult(rows=[{"id": 1}], latency_seconds=0.0)
        score = ExpectationSuiteScorer().score(case, _OUTPUT, result)
        assert score.passed is True
        assert score.explanation is None
        assert [o.kind for o in score.outcomes] == ["row_count", "column_presence", "unique", "not_null"]
        assert all(o.passed for o in score.outcomes)

    def test_satisfies_scorer_protocol(self) -> None:
        assert isinstance(ExpectationSuiteScorer(), Scorer)
