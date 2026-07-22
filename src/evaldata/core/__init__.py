"""Core orchestration: the runner, the pytest-facing `assert_eval`, and `run_benchmark`."""

from evaldata.core.runner import (
    BenchmarkSummary,
    CaseEvaluation,
    ExecutedCaseEvaluation,
    SolverFailedCaseEvaluation,
    assert_eval,
    evaluate_case,
    run_benchmark,
)

__all__ = [
    "BenchmarkSummary",
    "CaseEvaluation",
    "ExecutedCaseEvaluation",
    "SolverFailedCaseEvaluation",
    "assert_eval",
    "evaluate_case",
    "run_benchmark",
]
