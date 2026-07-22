"""Unit tests for the run-level eval-outcome collector."""

import json

import pytest
from pydantic import TypeAdapter, ValidationError

from evaldata.reporting.collector import (
    CaseReport,
    ExecutionFailureCaseReport,
    PassedCaseReport,
    ScoredFailureCaseReport,
    SolverFailureCaseReport,
    clear,
    extend,
    record,
    reports,
    run_report_json,
)
from evaldata.types import ExecutionError, ResultSetDiff, ScoreResult, SolverError


@pytest.fixture(autouse=True)
def _clean_collector() -> None:
    clear()


@pytest.mark.unit
class TestCollector:
    def test_record_and_reports_round_trip(self) -> None:
        report = PassedCaseReport(id="c1", input="q")
        record(report)
        assert reports() == [report]

    def test_reports_returns_a_snapshot_copy(self) -> None:
        record(PassedCaseReport(id="c1", input="q"))
        snapshot = reports()
        record(PassedCaseReport(id="c2", input="q"))
        assert len(snapshot) == 1  # snapshot is not a live view

    def test_clear_empties_the_accumulator(self) -> None:
        record(PassedCaseReport(id="c1", input="q"))
        clear()
        assert reports() == []

    def test_extend_appends_in_order(self) -> None:
        record(PassedCaseReport(id="c1", input="q"))
        extend(
            [
                ScoredFailureCaseReport(id="c2", input="q", scores=[ScoreResult(scorer="s", verdict="fail")]),
                PassedCaseReport(id="c3", input="q"),
            ]
        )
        assert [r.id for r in reports()] == ["c1", "c2", "c3"]

    def test_passing_report_rejects_non_passing_score(self) -> None:
        with pytest.raises(ValidationError, match="passing case report"):
            PassedCaseReport(id="bad", input="q", scores=[ScoreResult(scorer="s", verdict="fail")])

    def test_scored_failure_requires_non_passing_score(self) -> None:
        with pytest.raises(ValidationError, match="scored-failure"):
            ScoredFailureCaseReport(id="bad", input="q", scores=[ScoreResult(scorer="s", verdict="pass")])

    def test_execution_failure_round_trips(self) -> None:
        report = ExecutionFailureCaseReport(
            id="bad",
            input="q",
            error=ExecutionError(kind="query_failed", message="syntax error"),
        )
        restored = TypeAdapter(CaseReport).validate_json(report.model_dump_json())
        assert restored == report


@pytest.mark.unit
class TestRunReportJson:
    def test_counts_and_cases_serialized(self) -> None:
        diff = ResultSetDiff(expected_row_count=1, actual_row_count=0)
        record(PassedCaseReport(id="ok", input="q1"))
        record(
            ScoredFailureCaseReport(
                id="bad",
                input="q2",
                scores=[ScoreResult(scorer="result_set_equivalence", verdict="fail", diff=diff)],
            )
        )
        record(
            SolverFailureCaseReport(id="solver", input="q3", error=SolverError(kind="auth", message="invalid api key"))
        )

        payload = json.loads(run_report_json(reports()))
        assert payload["passed"] == 1
        assert payload["failed"] == 2
        assert [c["id"] for c in payload["cases"]] == ["ok", "bad", "solver"]
        # nested diff structure survives serialization
        assert payload["cases"][1]["scores"][0]["diff"]["expected_row_count"] == 1
        # the typed solver error serializes as a structured object; `cause` is excluded
        assert payload["cases"][2]["error"] == {"kind": "auth", "message": "invalid api key", "provider": None}
