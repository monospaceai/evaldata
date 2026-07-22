"""`Solver` Protocol: the contract for the AI system under test (`EvalCase` -> `SolverOutput`)."""

from typing import Protocol, runtime_checkable

from evaldata.types import EvalCase, SolverOutput, SolverSuccess


class SuccessfulSolver(Protocol):
    """Produces only successful SQL outputs."""

    def solve(self, case: EvalCase) -> SolverSuccess:
        """Produce executable SQL for `case`."""
        ...


@runtime_checkable
class Solver(Protocol):
    """Produces a `SolverOutput` for an `EvalCase`."""

    def solve(self, case: EvalCase) -> SolverOutput:
        """Produce output (for SQL solvers, the executable SQL) for `case`."""
        ...
