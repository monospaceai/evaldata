"""Run-level collection of eval outcomes, for the end-of-run summary and JSON artifact."""

import json
from collections.abc import Iterable, Sequence

from pydantic import BaseModel, ConfigDict, Field

from evaldata.types import ScoreResult, SolverError


class CaseReport(BaseModel):
    """One eval case's outcome: its identity, pass/fail, and per-scorer results (or a solver error)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    passed: bool
    scores: list[ScoreResult] = Field(default_factory=list)
    error: SolverError | None = None


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
