"""Unit tests for the Rich-backed terminal failure rendering."""

import pytest

from evaldata.reporting.collector import CaseReport
from evaldata.reporting.terminal import render_failure, render_solver_error, render_summary
from evaldata.types import (
    ColumnMismatch,
    EvalCase,
    ExecutionError,
    ExecutionResult,
    PlatformRef,
    ResultSetDiff,
    ScoreResult,
    SolverError,
    SolverOutput,
    TypeMismatch,
    UntypedResultSet,
)

_PLATFORM = PlatformRef(name="local", kind="duckdb")


def _case() -> EvalCase:
    return EvalCase(
        id="rock-count",
        input="How many tracks?",
        expected=UntypedResultSet(rows=[{"count": 99}]),
        platform=_PLATFORM,
    )


@pytest.mark.unit
class TestRenderFailure:
    def test_renders_case_id_and_sql_verbatim(self) -> None:
        out = SolverOutput(output="SELECT count(*) AS count FROM tracks WHERE genre = 'Rock'")
        result = ExecutionResult(rows=[{"count": 2}], latency_seconds=0.0)
        diff = ResultSetDiff(
            expected_row_count=1,
            actual_row_count=1,
            missing_row_count=1,
            extra_row_count=1,
            sample_missing_rows=[{"count": 99}],
            sample_extra_rows=[{"count": 2}],
        )
        score = ScoreResult(scorer="result_set_equivalence", passed=False, diff=diff)
        msg = render_failure(_case(), out, result, [score])
        assert "rock-count" in msg
        # SQL is surfaced verbatim — not soft-wrapped by the renderer
        assert "SELECT count(*) AS count FROM tracks WHERE genre = 'Rock'" in msg
        assert "result-set diff" in msg
        assert "missing rows" in msg and "extra rows" in msg
        assert "99" in msg and "2" in msg

    def test_renders_column_and_type_diffs(self) -> None:
        out = SolverOutput(output="SELECT 1")
        result = ExecutionResult(rows=[], latency_seconds=0.0)
        diff = ResultSetDiff(
            expected_row_count=1,
            actual_row_count=0,
            missing_columns=["amount"],
            unexpected_columns=["total"],
            column_order_mismatch=True,
            type_mismatches=[TypeMismatch(column="ts", expected="TIMESTAMP", actual="DATE")],
        )
        score = ScoreResult(scorer="result_set_equivalence", passed=False, diff=diff)
        msg = render_failure(_case(), out, result, [score])
        assert "missing columns" in msg and "amount" in msg
        assert "unexpected columns" in msg and "total" in msg
        assert "column order differs" in msg
        assert "type mismatches" in msg
        assert "TIMESTAMP" in msg and "DATE" in msg

    def test_renders_column_mismatches(self) -> None:
        out = SolverOutput(output="SELECT 1")
        result = ExecutionResult(rows=[{"amount": 5}], latency_seconds=0.0)
        diff = ResultSetDiff(
            expected_row_count=3,
            actual_row_count=3,
            column_mismatches=[
                ColumnMismatch(column="amount", unexpected_count=2),
                ColumnMismatch(column="status", unexpected_count=1),
            ],
        )
        score = ScoreResult(scorer="result_set_equivalence", passed=False, diff=diff)
        msg = render_failure(_case(), out, result, [score])
        assert "column mismatches" in msg
        assert "amount" in msg and "status" in msg
        assert "2" in msg and "1" in msg

    def test_bracketed_type_strings_survive(self) -> None:
        # Array/list-style values must not be eaten by Rich console markup ([...] tags).
        out = SolverOutput(output="SELECT 1")
        result = ExecutionResult(rows=[], latency_seconds=0.0)
        diff = ResultSetDiff(
            expected_row_count=1,
            actual_row_count=1,
            type_mismatches=[TypeMismatch(column="tags", expected="INTEGER[]", actual="VARCHAR[]")],
        )
        score = ScoreResult(scorer="result_set_equivalence", passed=False, diff=diff)
        msg = render_failure(_case(), out, result, [score])
        assert "INTEGER[]" in msg and "VARCHAR[]" in msg

    def test_renders_execution_error(self) -> None:
        out = SolverOutput(output="SELECT * FROM nope")
        result = ExecutionResult(
            rows=[], latency_seconds=0.0, error=ExecutionError(kind="query_failed", message="table nope does not exist")
        )
        score = ScoreResult(scorer="result_set_equivalence", passed=False, explanation="query failed")
        msg = render_failure(_case(), out, result, [score])
        assert "execution error: table nope does not exist" in msg


@pytest.mark.unit
class TestRenderSolverError:
    def test_renders_kind_and_message(self) -> None:
        error = SolverError(kind="auth", message="invalid api key", provider="openai")
        msg = render_solver_error(_case(), error)
        assert "rock-count" in msg
        assert "auth" in msg
        assert "invalid api key" in msg


@pytest.mark.unit
class TestRenderSummary:
    def test_rows_results_and_tally(self) -> None:
        summary = render_summary(
            [
                CaseReport(id="ok", input="q", passed=True),
                CaseReport(
                    id="bad",
                    input="q",
                    passed=False,
                    scores=[ScoreResult(scorer="result_set_equivalence", passed=False)],
                ),
            ]
        )
        assert "ok" in summary and "PASS" in summary
        assert "bad" in summary and "FAIL" in summary
        assert "result_set_equivalence" in summary  # failed-scorer name in the detail cell
        assert "1 passed, 1 failed" in summary

    def test_solver_error_shown_in_detail(self) -> None:
        summary = render_summary(
            [CaseReport(id="x", input="q", passed=False, error=SolverError(kind="auth", message="invalid api key"))]
        )
        assert "solver error [auth]" in summary
