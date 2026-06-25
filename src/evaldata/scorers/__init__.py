"""Scorers: pluggable pass/fail checks.

Ships `ResultSetEquivalence`, `ExpectationSuiteScorer`, `SemanticEquivalence`, and the
litellm-backed `LlmJudge`, plus the `FirstDecisive` combinator and the `query_equivalence`
composition it powers.
"""

from typing import TYPE_CHECKING, Any

from evaldata.scorers.base import Scorer
from evaldata.scorers.combinators import FirstDecisive
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.expectation_suite import ExpectationSuiteScorer
from evaldata.scorers.query import QueryRunner, ScalarResult
from evaldata.scorers.query_equivalence import query_equivalence
from evaldata.scorers.result_set_equivalence import ResultSetEquivalence
from evaldata.scorers.semantic_equivalence import (
    AstEquivalence,
    EquivalenceCheck,
    SemanticEquivalence,
    default_equivalence_checks,
)

if TYPE_CHECKING:
    from evaldata.scorers.llm_judge import LlmJudge

__all__ = [
    "AstEquivalence",
    "EquivalenceCheck",
    "ExpectationSuiteScorer",
    "FirstDecisive",
    "LlmJudge",
    "QueryRunner",
    "ResultSetEquivalence",
    "ScalarResult",
    "ScoreContext",
    "Scorer",
    "SemanticEquivalence",
    "default_equivalence_checks",
    "query_equivalence",
]


def __getattr__(name: str) -> Any:
    if name == "LlmJudge":
        try:
            from evaldata.scorers.llm_judge import LlmJudge
        except ImportError as e:
            msg = "LlmJudge requires the 'litellm' extra: install evaldata[litellm]"
            raise ImportError(msg) from e
        return LlmJudge
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "LlmJudge"])
