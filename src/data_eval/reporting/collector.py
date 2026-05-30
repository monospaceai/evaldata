"""Run-level collection of eval outcomes, for the end-of-run summary and JSON artifact.

``assert_eval`` is a plain function called inside a test body — it has no access to the
pytest ``item``/``request``, so it can't use ``record_property``. Following pytest-check's
precedent, it instead appends a ``CaseReport`` to a module-level accumulator here (for both
passing and failing cases); the pytest plugin reads the accumulation in
``pytest_terminal_summary`` (the Rich rollup table) and ``pytest_sessionfinish`` (the JSON
artifact). The accumulator lives for the process; the plugin never clears it (avoiding
ordering hazards between the summary and finish hooks) — process exit discards it, and the
framework's own test suite clears it per-test for isolation.

``CaseReport`` is the JSON artifact's schema. It lives here, with the reporting logic,
rather than in ``types.py`` (the core domain model) — a deliberate, easily-reversed split.

Known limitation: the accumulator is process-local, so under ``pytest-xdist`` the data
lives in the worker processes; the controller's summary/artifact aggregate only what the
controller itself ran. The plugin skips reporting on workers to avoid duplication.
"""

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from data_eval.types import ScoreResult


class CaseReport(BaseModel):
    """One eval case's outcome: its identity, pass/fail, and per-scorer results (or a solver error)."""

    model_config = ConfigDict(extra="forbid")

    id: str
    input: str
    passed: bool
    scores: list[ScoreResult] = Field(default_factory=list)
    error: str | None = None


_RUN: list[CaseReport] = []


def record(report: CaseReport) -> None:
    """Append a case outcome to the run accumulator."""
    _RUN.append(report)


def reports() -> list[CaseReport]:
    """Return a snapshot of the accumulated case outcomes."""
    return list(_RUN)


def clear() -> None:
    """Empty the run accumulator (used for per-test isolation in the framework's own suite)."""
    _RUN.clear()


def run_report_json(case_reports: Sequence[CaseReport]) -> str:
    """Serialize the run as a structured JSON artifact: pass/fail counts plus every case."""
    payload = {
        "passed": sum(1 for r in case_reports if r.passed),
        "failed": sum(1 for r in case_reports if not r.passed),
        "cases": [r.model_dump(mode="json") for r in case_reports],
    }
    return json.dumps(payload, indent=2)
