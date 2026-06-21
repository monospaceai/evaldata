"""Scorers: pluggable pass/fail checks.

Ships `ResultSetEquivalence`, `ExpectationSuiteScorer`, and `SemanticEquivalence`.
"""

from evaldata.scorers.base import Scorer
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.expectation_suite import ExpectationSuiteScorer
from evaldata.scorers.query import QueryRunner, ScalarResult
from evaldata.scorers.result_set_equivalence import ResultSetEquivalence
from evaldata.scorers.semantic_equivalence import (
    AstEquivalence,
    EquivalenceCheck,
    ExecutionEquivalence,
    SemanticEquivalence,
    default_equivalence_checks,
)

__all__ = [
    "AstEquivalence",
    "EquivalenceCheck",
    "ExecutionEquivalence",
    "ExpectationSuiteScorer",
    "QueryRunner",
    "ResultSetEquivalence",
    "ScalarResult",
    "ScoreContext",
    "Scorer",
    "SemanticEquivalence",
    "default_equivalence_checks",
]
