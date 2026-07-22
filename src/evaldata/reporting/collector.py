"""Run-level collection of eval outcomes, for the end-of-run summary and JSON artifact."""

import json
from collections.abc import Iterable, Sequence
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaldata.types import ExecutionError, ScoreResult, SolverError


class _CaseReportBase(BaseModel):
    """Identity shared by every case outcome."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    status: str

    @property
    def passed(self) -> bool:
        """Whether the case passed."""
        return self.status == "passed"


class PassedCaseReport(_CaseReportBase):
    """A passing case report."""

    status: Literal["passed"] = "passed"
    scores: list[ScoreResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def _all_scores_passed(self) -> "PassedCaseReport":
        """Reject non-passing scores on a passing report.

        Returns:
            The validated report.

        Raises:
            ValueError: If any scorer did not pass.
        """
        if any(not score.passed for score in self.scores):
            msg = "a passing case report cannot carry a failing or inconclusive score"
            raise ValueError(msg)
        return self


class ScoredFailureCaseReport(_CaseReportBase):
    """A case that did not pass scoring."""

    status: Literal["scored_failure"] = "scored_failure"
    scores: Annotated[list[ScoreResult], Field(min_length=1)]

    @model_validator(mode="after")
    def _has_non_passing_score(self) -> "ScoredFailureCaseReport":
        """Require a score that explains the failed report.

        Returns:
            The validated report.

        Raises:
            ValueError: If every score passed.
        """
        if all(score.passed for score in self.scores):
            msg = "a scored-failure case report requires a failing or inconclusive score"
            raise ValueError(msg)
        return self


class SolverFailureCaseReport(_CaseReportBase):
    """A case that failed before execution because the solver failed."""

    status: Literal["solver_failure"] = "solver_failure"
    error: SolverError


class ExecutionFailureCaseReport(_CaseReportBase):
    """A case whose generated SQL failed to execute."""

    status: Literal["execution_failure"] = "execution_failure"
    error: ExecutionError
    scores: list[ScoreResult] = Field(default_factory=list)


CaseReport: TypeAlias = Annotated[
    PassedCaseReport | ScoredFailureCaseReport | SolverFailureCaseReport | ExecutionFailureCaseReport,
    Field(discriminator="status"),
]


_RUN: list[CaseReport] = []


def record(report: CaseReport) -> None:
    """Append a case outcome to the run accumulator."""
    _RUN.append(report)


def extend(case_reports: Iterable[CaseReport]) -> None:
    """Append several case outcomes to the run accumulator.

    Args:
        case_reports: The case outcomes to append, in order.
    """
    _RUN.extend(case_reports)


def reports() -> list[CaseReport]:
    """Return a snapshot of the accumulated case outcomes."""
    return list(_RUN)


def clear() -> None:
    """Empty the run accumulator (used for per-test isolation in the framework's own suite)."""
    _RUN.clear()


def run_report_json(case_reports: Sequence[CaseReport]) -> str:
    """Serialize the run as a structured JSON artifact: pass/fail counts plus every case.

    Args:
        case_reports: The accumulated case outcomes to serialize.

    Returns:
        A JSON string with `passed`/`failed` counts and a `cases` array.
    """
    payload = {
        "passed": sum(1 for r in case_reports if r.passed),
        "failed": sum(1 for r in case_reports if not r.passed),
        "cases": [r.model_dump(mode="json") for r in case_reports],
    }
    return json.dumps(payload, indent=2)
