"""dataeval — AI evals framework for data and analytics engineering teams."""

from typing import TYPE_CHECKING, Any

from dataeval.core import assert_eval
from dataeval.loaders import eval_case
from dataeval.scorers import ExpectationSuiteScorer, ResultSetEquivalence
from dataeval.solvers import CallableSolver
from dataeval.types import EvalCase, PlatformRef

if TYPE_CHECKING:
    from dataeval.solvers import PromptSolver as PromptSolver

__all__ = [
    "CallableSolver",
    "EvalCase",
    "ExpectationSuiteScorer",
    "PlatformRef",
    "ResultSetEquivalence",
    "assert_eval",
    "eval_case",
]


def __getattr__(name: str) -> Any:
    if name == "PromptSolver":
        from dataeval.solvers import PromptSolver

        return PromptSolver
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "PromptSolver"])
