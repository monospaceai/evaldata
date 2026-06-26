"""evaldata — AI evals framework for data and analytics engineering teams."""

from typing import TYPE_CHECKING, Any

from evaldata.core import assert_eval
from evaldata.llm import Llm
from evaldata.loaders import eval_case
from evaldata.scorers import (
    JUDGE_INSTRUCTION,
    ExpectationSuiteScorer,
    FirstDecisive,
    JudgeExample,
    LlmJudge,
    ResultSetEquivalence,
    RubricBand,
    SemanticEquivalence,
    query_equivalence,
)
from evaldata.solvers import CallableSolver, PromptSolver
from evaldata.types import EvalCase, PlatformRef

if TYPE_CHECKING:
    from evaldata.llm import LiteLlm

__all__ = [
    "JUDGE_INSTRUCTION",
    "CallableSolver",
    "EvalCase",
    "ExpectationSuiteScorer",
    "FirstDecisive",
    "JudgeExample",
    "LiteLlm",
    "Llm",
    "LlmJudge",
    "PlatformRef",
    "PromptSolver",
    "ResultSetEquivalence",
    "RubricBand",
    "SemanticEquivalence",
    "assert_eval",
    "eval_case",
    "query_equivalence",
]


def __getattr__(name: str) -> Any:
    if name == "LiteLlm":
        from evaldata.llm import LiteLlm

        return LiteLlm
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "LiteLlm"])
