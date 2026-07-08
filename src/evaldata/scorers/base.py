"""`Scorer` Protocol: pluggable pass/fail check over an executed result."""

from typing import Protocol, runtime_checkable

from evaldata.scorers.context import ScoreContext
from evaldata.types import EvalCase, ExecutionResult, ScoreResult, SolverOutput


@runtime_checkable
class Scorer(Protocol):
    """Produces a `ScoreResult` from a case, its solver output, and the execution result."""

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Decide pass/fail with diagnostics for `case` given `output`, `result`, and `context`."""
        ...


def misconfigured(scorer: str, expected: object, requirement: str) -> ScoreResult:
    """Build an inconclusive result flagging a scorer paired with the wrong `expected` kind.

    The result carries `metadata["scorer_misconfigured"] = True` and an explanation naming the
    scorer, what it requires, and the `expected` kind it got, so the mismatch surfaces in the
    case report and terminal rendering rather than silently deciding the case.

    Args:
        scorer: The scorer's name.
        expected: The `case.expected` the scorer was given.
        requirement: A phrase describing what the scorer requires (e.g. `"a GoldQuery"`).

    Returns:
        An inconclusive `ScoreResult` describing the misconfiguration.
    """
    return ScoreResult(
        scorer=scorer,
        verdict="inconclusive",
        explanation=f"{scorer} requires {requirement}; got {type(expected).__name__}",
        metadata={"scorer_misconfigured": True},
    )
