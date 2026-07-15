"""`SqlEquivalence`: a pydantic-evals `Evaluator` that scores generated SQL by execution."""

import threading
from dataclasses import dataclass
from typing import Any

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from evaldata.core.runner import evaluate_case
from evaldata.scorers.equivalence_presets import observed_equivalence
from evaldata.solvers.callable import CallableSolver
from evaldata.types import (
    EvalCase,
    Expected,
    GoldQuery,
    PlatformRef,
    ResultSetDiff,
    ScoreResult,
    TypedResultSet,
    UntypedResultSet,
)

# Serialises scoring across cases: pydantic-evals runs cases concurrently, but a resolved
# platform adapter's connection is not thread-safe.
_SCORE_LOCK = threading.Lock()

_ACCEPTED_EXPECTED = "a str (gold SQL), GoldQuery, UntypedResultSet, or TypedResultSet"


@dataclass
class SqlEquivalence(Evaluator[Any, str, Any]):
    """Score generated SQL by executing it against a warehouse and checking equivalence.

    Reads the case's generated SQL from `ctx.output` and its reference from
    `ctx.expected_output` (a gold-SQL `str`, or a `GoldQuery`/`UntypedResultSet`/
    `TypedResultSet`), runs both against `platform`, and returns an `EvaluationReason` whose
    `value` is the pass/fail and whose `reason` explains it. Invalid generated SQL scores as a
    failure rather than raising; a missing or unusable case contract raises `ValueError`.

    Scoring is serialised across concurrently-run cases because the platform adapter holds a
    single, non-thread-safe connection. For parallelism, shard cases across distinct
    `PlatformRef` names or use evaldata's benchmark runner.

    Attributes:
        platform: The platform to execute the generated and reference SQL against.
    """

    platform: PlatformRef

    def __post_init__(self) -> None:
        """Build the equivalence scorer once, off the serialized dataclass fields."""
        self._scorer = observed_equivalence()

    def evaluate(self, ctx: EvaluatorContext[Any, str, Any]) -> EvaluationReason:
        """Score `ctx.output` against `ctx.expected_output` by executing both on `platform`.

        Args:
            ctx: The evaluation context; `output` is the generated SQL and `expected_output`
                is the reference (gold SQL `str` or an evaldata result-set/gold expectation).

        Returns:
            An `EvaluationReason` whose `value` is `True` when the queries are equivalent and
            whose `reason` states the verdict and any diff.

        Raises:
            ValueError: If `output` is not a non-empty string, or if `expected_output` is not
                one of the accepted kinds.
        """
        if not isinstance(ctx.output, str) or not ctx.output.strip():
            msg = "SqlEquivalence requires ctx.output to be a non-empty SQL string"
            raise ValueError(msg)
        expected = _map_expected(ctx.expected_output)

        question = ctx.inputs if isinstance(ctx.inputs, str) and ctx.inputs.strip() else (ctx.name or "generated query")
        case = EvalCase(
            id=ctx.name or "pydantic-evals-case",
            input=question,
            expected=expected,
            platform=self.platform,
        )
        solver = CallableSolver(lambda _case: ctx.output)
        with _SCORE_LOCK:
            evaluation = evaluate_case(case, solver, scorers=[self._scorer])
        score = evaluation.report.scores[0]
        return EvaluationReason(value=score.passed, reason=_reason(score))


def _map_expected(expected_output: Any) -> Expected:
    """Map a case's `expected_output` onto an evaldata `Expected`.

    Args:
        expected_output: A gold-SQL `str`, or a `GoldQuery`/`UntypedResultSet`/`TypedResultSet`.

    Returns:
        The corresponding `Expected`: a `str` becomes a `GoldQuery`; the accepted instances pass
        through unchanged.

    Raises:
        ValueError: If `expected_output` is any other kind.
    """
    if isinstance(expected_output, str):
        return GoldQuery(sql=expected_output)
    if isinstance(expected_output, GoldQuery | UntypedResultSet | TypedResultSet):
        return expected_output
    msg = (
        f"SqlEquivalence requires ctx.expected_output to be {_ACCEPTED_EXPECTED}; got {type(expected_output).__name__}"
    )
    raise ValueError(msg)


def _reason(score: ScoreResult) -> str:
    """Compose a human-readable reason from a score's verdict, diff, and explanation.

    Args:
        score: The scorer result to describe.

    Returns:
        A non-empty reason stating the verdict, any result-set diff, and the scorer's own
        explanation when present.
    """
    parts = ["queries are equivalent" if score.passed else "queries are not equivalent"]
    if score.diff is not None:
        parts.append(_summarize_diff(score.diff))
    if score.explanation:
        parts.append(score.explanation)
    return "; ".join(parts)


def _summarize_diff(diff: ResultSetDiff) -> str:
    """Summarise a result-set diff's row-count mismatch in one line.

    Args:
        diff: The structured difference between the actual and expected result sets.

    Returns:
        A one-line summary of the expected/actual row counts and the missing/extra counts.
    """
    return (
        f"expected {diff.expected_row_count} row(s), got {diff.actual_row_count} "
        f"({diff.missing_row_count} missing, {diff.extra_row_count} extra)"
    )
