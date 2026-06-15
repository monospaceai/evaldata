"""Unit tests for the run-level eval-outcome collector."""

import json

import pytest

from dataeval.reporting.collector import CaseReport, clear, extend, record, reports, run_report_json
from dataeval.types import ResultSetDiff, ScoreResult


@pytest.fixture(autouse=True)
def _clean_collector() -> None:
    clear()


@pytest.mark.unit
class TestCollector:
    def test_record_and_reports_round_trip(self) -> None:
        report = CaseReport(id="c1", input="q", passed=True)
        record(report)
        assert reports() == [report]

    def test_reports_returns_a_snapshot_copy(self) -> None:
        record(CaseReport(id="c1", input="q", passed=True))
        snapshot = reports()
        record(CaseReport(id="c2", input="q", passed=True))
        assert len(snapshot) == 1  # snapshot is not a live view

    def test_clear_empties_the_accumulator(self) -> None:
        record(CaseReport(id="c1", input="q", passed=True))
        clear()
        assert reports() == []

    def test_extend_appends_in_order(self) -> None:
        record(CaseReport(id="c1", input="q", passed=True))
        extend(
            [
                CaseReport(id="c2", input="q", passed=False),
                CaseReport(id="c3", input="q", passed=True),
            ]
        )
        assert [r.id for r in reports()] == ["c1", "c2", "c3"]


@pytest.mark.unit
class TestRunReportJson:
    def test_counts_and_cases_serialized(self) -> None:
        diff = ResultSetDiff(expected_row_count=1, actual_row_count=0)
        record(CaseReport(id="ok", input="q1", passed=True))
        record(
            CaseReport(
                id="bad",
                input="q2",
                passed=False,
                scores=[ScoreResult(scorer="result_set_equivalence", passed=False, diff=diff)],
            )
        )
        record(CaseReport(id="solver", input="q3", passed=False, error="solver error [auth]"))

        payload = json.loads(run_report_json(reports()))
        assert payload["passed"] == 1
        assert payload["failed"] == 2
        assert [c["id"] for c in payload["cases"]] == ["ok", "bad", "solver"]
        # nested diff structure survives serialization
        assert payload["cases"][1]["scores"][0]["diff"]["expected_row_count"] == 1
        assert payload["cases"][2]["error"] == "solver error [auth]"
