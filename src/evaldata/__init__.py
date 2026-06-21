"""evaldata — AI evals framework for data and analytics engineering teams."""

from typing import TYPE_CHECKING, Any

from evaldata.core import assert_eval
from evaldata.loaders import eval_case
from evaldata.scorers import ExpectationSuiteScorer, ResultSetEquivalence, SemanticEquivalence
from evaldata.solvers import CallableSolver
from evaldata.types import EvalCase, PlatformRef

if TYPE_CHECKING:
    from evaldata.solvers import PromptSolver as PromptSolver

__all__ = [
    "CallableSolver",
    "EvalCase",
    "ExpectationSuiteScorer",
    "PlatformRef",
    "ResultSetEquivalence",
    "SemanticEquivalence",
    "assert_eval",
    "eval_case",
]


def __getattr__(name: str) -> Any:
    if name == "PromptSolver":
        from evaldata.solvers import PromptSolver

        return PromptSolver
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "PromptSolver"])
