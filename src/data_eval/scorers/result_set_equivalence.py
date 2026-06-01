"""`ResultSetEquivalence`: result-set scorer wrapping the equivalence engine."""

from data_eval.equivalence import TypedResultSet, UntypedResultSet, compare
from data_eval.types import (
    EvalCase,
    ExecutionResult,
    ExpectedResultSet,
    ScoreResult,
    SolverOutput,
)

SCORER_NAME = "result_set_equivalence"


class ResultSetEquivalence:
    """Scores a case by comparing its executed result set against its `ExpectedResultSet`."""

    def score(self, case: EvalCase, output: SolverOutput, result: ExecutionResult) -> ScoreResult:
        """Compare `result` against `case.expected`; pass iff the engine finds them equivalent.

        Args:
            case: The eval case, carrying the expected result set and platform.
            output: The solver output (part of the `Scorer` protocol; unused here).
            result: The executed result to compare against the expectation.

        Returns:
            A `ScoreResult` that passes when the result set matches the expectation; a
            failed query yields a failing result with an explanation.

        Raises:
            TypeError: If `case.expected` is not an `ExpectedResultSet`.
        """
        expected = case.expected
        if not isinstance(expected, ExpectedResultSet):
            msg = f"ResultSetEquivalence requires an ExpectedResultSet; got {type(expected).__name__}"
            raise TypeError(msg)

        if result.error is not None:
            return ScoreResult(
                scorer=SCORER_NAME,
                passed=False,
                explanation=f"query execution failed: {result.error}",
            )

        if expected.schema_ is not None and result.schema_ is not None:
            diff = compare(
                TypedResultSet(rows=result.rows, schema=result.schema_),
                TypedResultSet(rows=expected.rows, schema=expected.schema_),
                case.comparison,
            )
        else:
            diff = compare(
                UntypedResultSet(rows=result.rows),
                UntypedResultSet(rows=expected.rows),
                case.comparison,
            )

        return ScoreResult(scorer=SCORER_NAME, passed=diff is None, diff=diff)
