"""evaldata — AI evals framework for data and analytics engineering teams."""

from typing import TYPE_CHECKING, Any

from evaldata.core import assert_eval
from evaldata.loaders import eval_case
from evaldata.scorers import (
    ExpectationSuiteScorer,
    FirstDecisive,
    ResultSetEquivalence,
    SemanticEquivalence,
    query_equivalence,
)
from evaldata.solvers import CallableSolver
from evaldata.types import EvalCase, PlatformRef

if TYPE_CHECKING:
    from evaldata.scorers import LlmJudge as LlmJudge
    from evaldata.solvers import PromptSolver as PromptSolver

__all__ = [
    "CallableSolver",
    "EvalCase",
    "ExpectationSuiteScorer",
    "FirstDecisive",
    "LlmJudge",
    "PlatformRef",
    "ResultSetEquivalence",
    "SemanticEquivalence",
    "assert_eval",
    "eval_case",
    "query_equivalence",
]

_LAZY = {"PromptSolver": "evaldata.solvers", "LlmJudge": "evaldata.scorers"}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is not None:
        import importlib

        return getattr(importlib.import_module(module), name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])
