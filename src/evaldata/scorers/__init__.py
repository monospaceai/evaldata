"""Scorers: pluggable pass/fail checks.

Ships `ResultSetEquivalence`, `ExpectationSuiteScorer`, `SemanticEquivalence`, and the
LLM-as-judge `LlmJudge`, plus the `FirstDecisive` combinator and the equivalence-preset
compositions it powers.
"""

from evaldata.scorers.base import Scorer
from evaldata.scorers.combinators import FirstDecisive
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.equivalence_presets import (
    judged_equivalence,
    observed_equivalence,
    sql_equivalence_judge,
)
from evaldata.scorers.execution_accuracy import ExecutionAccuracy
from evaldata.scorers.expectation_suite import ExpectationSuiteScorer
from evaldata.scorers.llm_judge import JUDGE_INSTRUCTION, JudgeExample, LlmJudge, RubricBand
from evaldata.scorers.query import QueryRunner, ScalarFailure, ScalarResult, ScalarSuccess
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
    "ExecutionAccuracy",
    "ExpectationSuiteScorer",
    "FirstDecisive",
    "JudgeExample",
    "LlmJudge",
    "QueryRunner",
    "RubricBand",
    "ResultSetEquivalence",
    "ScalarResult",
    "ScalarFailure",
    "ScalarSuccess",
    "ScoreContext",
    "Scorer",
    "SemanticEquivalence",
    "default_equivalence_checks",
    "judged_equivalence",
    "observed_equivalence",
    "sql_equivalence_judge",
]
