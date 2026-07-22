"""`CallableSolver`: adapt a plain function into a `Solver` (no-LLM, deterministic)."""

from collections.abc import Callable

from evaldata.types import EvalCase, SolverSuccess, Sql


class CallableSolver:
    """Wraps a function `(EvalCase) -> sql` as a `Solver`."""

    def __init__(self, fn: Callable[[EvalCase], str]) -> None:
        """Store the SQL-producing function `fn`."""
        self._fn = fn

    def solve(self, case: EvalCase) -> SolverSuccess:
        """Call the wrapped function and return its SQL as `SolverSuccess.output`.

        Args:
            case: The eval case to solve.

        Returns:
            A `SolverSuccess` carrying the SQL produced by the wrapped function.
        """
        return SolverSuccess(output=Sql(self._fn(case)))
