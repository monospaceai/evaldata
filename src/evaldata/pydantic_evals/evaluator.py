"""`SqlEquivalence`: a pydantic-evals `Evaluator` that scores generated SQL by execution."""

from dataclasses import dataclass
from typing import Any

import anyio.to_thread
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

_ACCEPTED_EXPECTED = "a str (gold SQL), GoldQuery, UntypedResultSet, or TypedResultSet"


@dataclass
class SqlEquivalence(Evaluator[Any, str, Any]):
    """Score generated SQL by executing it against a warehouse and checking equivalence.

    Reads the case's generated SQL from `ctx.output` and its reference from
    `ctx.expected_output` (a gold-SQL `str`, or a `GoldQuery`/`UntypedResultSet`/
    `TypedResultSet`), runs both against `platform`, and returns an `EvaluationReason` whose
    `value` is the pass/fail and whose `reason` explains it. Invalid generated SQL scores as a
    failure rather than raising; a missing or unusable case contract raises `ValueError`.

    Runs safely under Pydantic Evals concurrency: each case acquires its own platform session
    for the duration of scoring, so `max_concurrency` parallelizes warehouse execution up to
    the platform's per-name pool size.

    Attributes:
        platform: The platform to execute the generated and reference SQL against.
    """

    platform: PlatformRef

    def __post_init__(self) -> None:
        """Build the equivalence scorer once, off the serialized dataclass fields."""
        self._scorer = observed_equivalence()

    async def evaluate_async(self, ctx: EvaluatorContext[Any, str, Any]) -> EvaluationReason:
        """Score `ctx` on a worker thread so Pydantic Evals runs cases in parallel.

        Args:
            ctx: The evaluation context passed straight to `evaluate`.

        Returns:
            The `EvaluationReason` produced by `evaluate`.
        """
        return await anyio.to_thread.run_sync(self.evaluate, ctx)

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
        evaluation = evaluate_case(case, solver, scorers=[self._scorer])
        report = evaluation.report
        score = report.scores[0]
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
    """Summarise a result-set diff's row, column, type, and value mismatches.

    Args:
        diff: The structured difference between the actual and expected result sets.

    Returns:
        A one-line summary covering the row counts and any column, column-order, type, or
        per-column value differences the diff records.
    """
    parts = [
        f"expected {diff.expected_row_count} row(s), got {diff.actual_row_count} "
        f"({diff.missing_row_count} missing, {diff.extra_row_count} extra)"
    ]
    if diff.missing_columns:
        parts.append(f"missing columns: {', '.join(diff.missing_columns)}")
    if diff.unexpected_columns:
        parts.append(f"unexpected columns: {', '.join(diff.unexpected_columns)}")
    if diff.column_order_mismatch:
        parts.append("column order differs")
    if diff.type_mismatches:
        types = ", ".join(f"{m.column} expected {m.expected}, got {m.actual}" for m in diff.type_mismatches)
        parts.append(f"type mismatches: {types}")
    if diff.column_mismatches:
        values = ", ".join(f"{m.column} ({m.unexpected_count})" for m in diff.column_mismatches)
        parts.append(f"value mismatches: {values}")
    return "; ".join(parts)
