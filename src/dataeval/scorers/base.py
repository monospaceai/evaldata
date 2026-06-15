"""`Scorer` Protocol: pluggable pass/fail check over an executed result."""

from typing import Protocol, runtime_checkable

from dataeval.scorers.context import ScoreContext
from dataeval.types import EvalCase, ExecutionResult, ScoreResult, SolverOutput


@runtime_checkable
class Scorer(Protocol):
    """Produces a `ScoreResult` from a case, its solver output, and the execution result."""

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Decide pass/fail with diagnostics for `case` given `output`, `result`, and `context`."""
        ...
