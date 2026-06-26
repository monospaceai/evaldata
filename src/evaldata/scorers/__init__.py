"""Scorers: pluggable pass/fail checks.

Ships `ResultSetEquivalence`, `ExpectationSuiteScorer`, `SemanticEquivalence`, and the
LLM-as-judge `LlmJudge`, plus the `FirstDecisive` combinator and the `query_equivalence`
composition it powers.
"""

from evaldata.scorers.base import Scorer
from evaldata.scorers.combinators import FirstDecisive
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.expectation_suite import ExpectationSuiteScorer
from evaldata.scorers.llm_judge import JUDGE_INSTRUCTION, JudgeExample, LlmJudge, RubricBand
from evaldata.scorers.query import QueryRunner, ScalarResult
from evaldata.scorers.query_equivalence import query_equivalence
from evaldata.scorers.result_set_equivalence import ResultSetEquivalence
from evaldata.scorers.semantic_equivalence import (
    AstEquivalence,
    EquivalenceCheck,
    SemanticEquivalence,
    default_equivalence_checks,
)

__all__ = [
    "JUDGE_INSTRUCTION",
    "AstEquivalence",
    "EquivalenceCheck",
    "ExpectationSuiteScorer",
    "FirstDecisive",
    "JudgeExample",
    "LlmJudge",
    "QueryRunner",
    "RubricBand",
    "ResultSetEquivalence",
    "ScalarResult",
    "ScoreContext",
    "Scorer",
    "SemanticEquivalence",
    "default_equivalence_checks",
    "query_equivalence",
]
