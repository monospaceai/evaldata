"""evaldata — AI evals framework for data and analytics engineering teams."""

from typing import TYPE_CHECKING, Any

from evaldata.core import BenchmarkSummary, assert_eval, run_benchmark
from evaldata.llm import Completion, Llm, StubLlm, TextCompletion, Usage
from evaldata.loaders import eval_case, load_bird, load_spider
from evaldata.platforms.registry import (
    bigquery_platform,
    databricks_platform,
    duckdb_platform,
    postgres_platform,
    snowflake_platform,
    sqlite_platform,
)
from evaldata.scorers import (
    JUDGE_INSTRUCTION,
    EquivalenceCheck,
    ExecutionAccuracy,
    ExpectationSuiteScorer,
    FirstDecisive,
    JudgeExample,
    LlmJudge,
    QueryRunner,
    ResultSetEquivalence,
    RubricBand,
    ScoreContext,
    Scorer,
    SemanticEquivalence,
    judged_equivalence,
    observed_equivalence,
    sql_equivalence_judge,
)
from evaldata.solvers import SCHEMA_PROMPT_TEMPLATE, CallableSolver, PromptSolver, Solver
from evaldata.types import EvalCase, PlatformRef

if TYPE_CHECKING:
    from evaldata.llm import LiteLlm

__all__ = [
    "JUDGE_INSTRUCTION",
    "SCHEMA_PROMPT_TEMPLATE",
    "BenchmarkSummary",
    "CallableSolver",
    "Completion",
    "EquivalenceCheck",
    "EvalCase",
    "ExecutionAccuracy",
    "ExpectationSuiteScorer",
    "FirstDecisive",
    "JudgeExample",
    "LiteLlm",
    "Llm",
    "LlmJudge",
    "PlatformRef",
    "PromptSolver",
    "QueryRunner",
    "ResultSetEquivalence",
    "RubricBand",
    "ScoreContext",
    "Scorer",
    "SemanticEquivalence",
    "Solver",
    "StubLlm",
    "TextCompletion",
    "Usage",
    "assert_eval",
    "bigquery_platform",
    "databricks_platform",
    "duckdb_platform",
    "eval_case",
    "judged_equivalence",
    "load_bird",
    "load_spider",
    "observed_equivalence",
    "postgres_platform",
    "run_benchmark",
    "snowflake_platform",
    "sql_equivalence_judge",
    "sqlite_platform",
]


def __getattr__(name: str) -> Any:
    if name == "LiteLlm":
        from evaldata.llm import LiteLlm

        return LiteLlm
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "LiteLlm"])
