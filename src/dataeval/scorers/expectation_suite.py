"""`ExpectationSuiteScorer`: evaluates an `ExpectationSuite` against an executed result."""

from typing import assert_never

from dataeval.scorers import sql
from dataeval.scorers.context import ScoreContext
from dataeval.scorers.query import QueryRunner
from dataeval.types import (
    ColumnPresenceExpectation,
    ColumnTypeExpectation,
    EvalCase,
    ExecutionResult,
    Expectation,
    ExpectationOutcome,
    ExpectationSuite,
    NotNullExpectation,
    RowCountExpectation,
    ScoreResult,
    SolverOutput,
    UniqueExpectation,
)

SCORER_NAME = "expectation_suite"


class ExpectationSuiteScorer:
    """Scores a case by checking its executed result against each `Expectation` in its suite."""

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Evaluate every expectation in the suite; pass iff all hold.

        Row-level checks (`row_count`, `not_null`, `unique`) are pushed into the platform as
        SQL over the model's query via `context.queries`; only counts and bounded failing-row
        samples are read back. Schema checks (`column_presence`, `column_type`) read the
        result's schema metadata and run no query.

        Args:
            case: The eval case, carrying the `ExpectationSuite`.
            output: The solver output (part of the `Scorer` protocol; unused here).
            result: The executed result; its schema is used by the schema checks.
            context: The score context, carrying the budget-aware `QueryRunner`.

        Returns:
            A `ScoreResult` that passes when all expectations hold. `outcomes` carries one
            `ExpectationOutcome` per expectation (passing and failing alike); on failure
            `explanation` lists each unmet expectation, derived from those outcomes. A failed
            model query yields a failing result with an explanation and no outcomes; a failed
            derived query fails only its own expectation's outcome.

        Raises:
            TypeError: If `case.expected` is not an `ExpectationSuite`.
        """
        expected = case.expected
        if not isinstance(expected, ExpectationSuite):
            msg = f"ExpectationSuiteScorer requires an ExpectationSuite; got {type(expected).__name__}"
            raise TypeError(msg)

        if result.error is not None:
            return ScoreResult(
                scorer=SCORER_NAME,
                passed=False,
                explanation=f"query execution failed: {result.error}",
            )

        outcomes = [_evaluate_one(e, result, context.queries) for e in expected.expectations]
        failures = [o for o in outcomes if not o.passed]
        if not failures:
            return ScoreResult(scorer=SCORER_NAME, passed=True, outcomes=outcomes)
        explanation = "\n".join([f"{len(failures)} expectation(s) failed:", *(f"  - {o.detail}" for o in failures)])
        return ScoreResult(scorer=SCORER_NAME, passed=False, outcomes=outcomes, explanation=explanation)


def _evaluate_one(expectation: Expectation, result: ExecutionResult, queries: QueryRunner) -> ExpectationOutcome:
    """Check one expectation against `result`, pushing row-level checks into the platform.

    Args:
        expectation: The expectation to check.
        result: The executed result (for schema checks and absent-column guards).
        queries: The budget-aware runner used to push row-level checks into the platform.

    Returns:
        An `ExpectationOutcome` recording pass/fail and the compared values; its `detail`
        holds a human-readable failure message, or `None` when the expectation holds.
    """
    model_sql = queries.model_sql
    dialect = queries.dialect
    match expectation:
        case RowCountExpectation():
            scalar = queries.scalar(sql.row_count(model_sql, dialect))
            if scalar.error is not None:
                return _query_error_outcome(expectation.kind, None, scalar.error)
            actual = scalar.value
            passed = actual == expectation.exact
            return ExpectationOutcome(
                kind=expectation.kind,
                passed=passed,
                expected=str(expectation.exact),
                actual=str(actual),
                detail=None if passed else f"row_count: expected {expectation.exact} rows, got {actual}",
            )
        case ColumnPresenceExpectation():
            present = _result_column_names(result)
            missing = [c for c in expectation.columns if c not in present]
            return ExpectationOutcome(
                kind=expectation.kind,
                passed=not missing,
                detail=None if not missing else f"column_presence: missing column(s) {missing}",
            )
        case ColumnTypeExpectation():
            expected_raw = expectation.expected_type.raw
            if result.schema_ is None:
                return ExpectationOutcome(
                    kind=expectation.kind,
                    passed=False,
                    column=expectation.column,
                    expected=expected_raw,
                    detail=f"column_type: no column schema available for column {expectation.column!r}",
                )
            types = dict(zip(result.schema_.names, result.schema_.types, strict=True))
            if expectation.column not in types:
                return ExpectationOutcome(
                    kind=expectation.kind,
                    passed=False,
                    column=expectation.column,
                    expected=expected_raw,
                    detail=f"column_type: column {expectation.column!r} not found in result",
                )
            actual_type = types[expectation.column]
            passed = actual_type == expectation.expected_type
            return ExpectationOutcome(
                kind=expectation.kind,
                passed=passed,
                column=expectation.column,
                expected=expected_raw,
                actual=actual_type.raw,
                detail=None
                if passed
                else f"column_type: column {expectation.column!r} expected type {expected_raw!r}, got {actual_type.raw!r}",
            )
        case NotNullExpectation():
            if expectation.column not in _result_column_names(result):
                return ExpectationOutcome(
                    kind=expectation.kind,
                    passed=False,
                    column=expectation.column,
                    detail=f"not_null: column {expectation.column!r} not found in result",
                )
            scalar = queries.scalar(sql.not_null_count(model_sql, expectation.column, dialect))
            if scalar.error is not None:
                return _query_error_outcome(expectation.kind, expectation.column, scalar.error)
            null_count = scalar.value
            if null_count == 0:
                return ExpectationOutcome(kind=expectation.kind, passed=True, column=expectation.column, count=0)
            sample = queries.run(sql.not_null_sample(model_sql, expectation.column, dialect))
            return ExpectationOutcome(
                kind=expectation.kind,
                passed=False,
                column=expectation.column,
                count=null_count,
                sample_rows=sample.rows,
                detail=f"not_null: column {expectation.column!r} has {null_count} NULL value(s)",
            )
        case UniqueExpectation():
            if expectation.column not in _result_column_names(result):
                return ExpectationOutcome(
                    kind=expectation.kind,
                    passed=False,
                    column=expectation.column,
                    detail=f"unique: column {expectation.column!r} not found in result",
                )
            scalar = queries.scalar(sql.unique_count(model_sql, expectation.column, dialect))
            if scalar.error is not None:
                return _query_error_outcome(expectation.kind, expectation.column, scalar.error)
            duplicate_count = scalar.value
            if duplicate_count == 0:
                return ExpectationOutcome(kind=expectation.kind, passed=True, column=expectation.column, count=0)
            sample = queries.run(sql.unique_sample(model_sql, expectation.column, dialect))
            return ExpectationOutcome(
                kind=expectation.kind,
                passed=False,
                column=expectation.column,
                count=duplicate_count,
                sample_rows=sample.rows,
                detail=f"unique: column {expectation.column!r} has {duplicate_count} duplicated value(s)",
            )
        case _:  # pragma: no cover - exhaustive over the Expectation union
            assert_never(expectation)


def _query_error_outcome(kind: str, column: str | None, error: str) -> ExpectationOutcome:
    """Build a failing outcome for an expectation whose derived query errored.

    Args:
        kind: The expectation's kind.
        column: The checked column, or `None` for column-less checks.
        error: The derived-query error message.

    Returns:
        A failing `ExpectationOutcome` carrying the error in `detail`.
    """
    return ExpectationOutcome(
        kind=kind,
        passed=False,
        column=column,
        detail=f"{kind}: query failed: {error}",
    )


def _result_column_names(result: ExecutionResult) -> list[str]:
    """Resolve the result's column names: the schema if present, else the first row's keys.

    Args:
        result: The executed result.

    Returns:
        The column names in order, or `[]` when neither a schema nor any rows are present.
    """
    if result.schema_ is not None:
        return result.schema_.names
    if result.rows:
        return list(result.rows[0].keys())
    return []
