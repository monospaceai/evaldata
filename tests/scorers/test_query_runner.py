"""Tests for `QueryRunner` — budget-aware derived-query execution and scalar read-back."""

import pytest

from dataeval.scorers import QueryRunner
from dataeval.types import ExecutionResult, Sql


class _RecordingAdapter:
    def __init__(self, results: list[ExecutionResult]) -> None:
        self.executed: list[str] = []
        self._results = list(results)

    def execute(self, sql: str) -> ExecutionResult:
        self.executed.append(sql)
        return self._results.pop(0)

    def cancel(self) -> None: ...

    def close(self) -> None: ...


def _runner(results: list[ExecutionResult], budget: float | None) -> tuple[QueryRunner, _RecordingAdapter]:
    adapter = _RecordingAdapter(results)
    return QueryRunner(adapter, Sql("SELECT 1"), "duckdb", budget), adapter


@pytest.mark.unit
class TestQueryRunner:
    def test_success_passes_result_through(self) -> None:
        result = ExecutionResult(rows=[{"n": 1}], latency_seconds=0.5)
        runner, adapter = _runner([result], None)
        out = runner.run(Sql("SELECT 1"))
        assert out is result
        assert adapter.executed == ["SELECT 1"]

    def test_model_sql_and_dialect_exposed(self) -> None:
        runner, _ = _runner([], None)
        assert runner.model_sql == "SELECT 1"
        assert runner.dialect == "duckdb"

    def test_underlying_error_returned_after_executing(self) -> None:
        result = ExecutionResult(rows=[], latency_seconds=0.0, error="boom")
        runner, adapter = _runner([result], None)
        out = runner.run(Sql("SELECT bad"))
        assert out.error == "boom"
        assert adapter.executed == ["SELECT bad"]

    def test_pool_decrements_across_calls(self) -> None:
        results = [
            ExecutionResult(rows=[], latency_seconds=2.0),
            ExecutionResult(rows=[], latency_seconds=2.0),
        ]
        runner, adapter = _runner(results, 5.0)
        runner.run(Sql("SELECT 1"))
        runner.run(Sql("SELECT 2"))
        assert len(adapter.executed) == 2

    def test_exhausted_pool_fails_fast(self) -> None:
        results = [
            ExecutionResult(rows=[], latency_seconds=1.0),
            ExecutionResult(rows=[], latency_seconds=1.0),
        ]
        runner, adapter = _runner(results, 1.0)
        runner.run(Sql("SELECT 1"))
        out = runner.run(Sql("SELECT 2"))
        assert out.error is not None
        assert "budget" in out.error
        assert len(adapter.executed) == 1

    def test_unbounded_pool_never_short_circuits(self) -> None:
        results = [
            ExecutionResult(rows=[], latency_seconds=10.0),
            ExecutionResult(rows=[], latency_seconds=10.0),
        ]
        runner, adapter = _runner(results, None)
        runner.run(Sql("SELECT 1"))
        runner.run(Sql("SELECT 2"))
        assert len(adapter.executed) == 2


@pytest.mark.unit
class TestScalar:
    def test_single_cell_success(self) -> None:
        runner, adapter = _runner([ExecutionResult(rows=[{"n": 3}], latency_seconds=0.2)], None)
        out = runner.scalar(Sql("SELECT count(*)"))
        assert out.error is None
        assert out.value == 3
        assert out.latency_seconds == 0.2
        assert adapter.executed == ["SELECT count(*)"]

    def test_underlying_error_propagated(self) -> None:
        runner, _ = _runner([ExecutionResult(rows=[], latency_seconds=0.1, error="boom")], None)
        out = runner.scalar(Sql("SELECT bad"))
        assert out.value is None
        assert out.error == "boom"
        assert out.latency_seconds == 0.1

    def test_zero_rows_is_error(self) -> None:
        runner, _ = _runner([ExecutionResult(rows=[], latency_seconds=0.0)], None)
        out = runner.scalar(Sql("SELECT 1 WHERE 1=0"))
        assert out.value is None
        assert out.error == "expected one row and one column"

    def test_multiple_rows_is_error(self) -> None:
        runner, _ = _runner([ExecutionResult(rows=[{"n": 1}, {"n": 2}], latency_seconds=0.0)], None)
        out = runner.scalar(Sql("SELECT n"))
        assert out.error == "expected one row and one column"

    def test_multiple_columns_is_error(self) -> None:
        runner, _ = _runner([ExecutionResult(rows=[{"a": 1, "b": 2}], latency_seconds=0.0)], None)
        out = runner.scalar(Sql("SELECT a, b"))
        assert out.error == "expected one row and one column"

    def test_draws_down_budget_pool(self) -> None:
        results = [
            ExecutionResult(rows=[{"n": 1}], latency_seconds=1.0),
            ExecutionResult(rows=[{"n": 1}], latency_seconds=1.0),
        ]
        runner, adapter = _runner(results, 1.0)
        runner.scalar(Sql("SELECT 1"))
        out = runner.scalar(Sql("SELECT 2"))
        assert out.error is not None
        assert "budget" in out.error
        assert len(adapter.executed) == 1
