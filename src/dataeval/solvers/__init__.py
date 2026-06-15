"""Solvers: wrappers for the AI system under test (`EvalCase` -> `SolverOutput`)."""

from typing import TYPE_CHECKING, Any

from dataeval.solvers.base import Solver
from dataeval.solvers.callable import CallableSolver

if TYPE_CHECKING:
    from dataeval.solvers.prompt import PromptSolver

__all__ = ["CallableSolver", "PromptSolver", "Solver"]


def __getattr__(name: str) -> Any:
    if name == "PromptSolver":
        try:
            from dataeval.solvers.prompt import PromptSolver
        except ImportError as e:
            msg = "PromptSolver requires the 'litellm' extra: install dataeval[litellm]"
            raise ImportError(msg) from e
        return PromptSolver
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "PromptSolver"])
