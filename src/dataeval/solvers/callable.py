"""`CallableSolver`: adapt a plain function into a `Solver` (no-LLM, deterministic)."""

from collections.abc import Callable

from dataeval.types import EvalCase, SolverOutput, Sql


class CallableSolver:
    """Wraps a function `(EvalCase) -> sql` as a `Solver`."""

    def __init__(self, fn: Callable[[EvalCase], str]) -> None:
        """Store the SQL-producing function `fn`."""
        self._fn = fn

    def solve(self, case: EvalCase) -> SolverOutput:
        """Call the wrapped function and return its SQL as `SolverOutput.output`.

        Args:
            case: The eval case to solve.

        Returns:
            A `SolverOutput` carrying the SQL produced by the wrapped function.
        """
        return SolverOutput(output=Sql(self._fn(case)))
