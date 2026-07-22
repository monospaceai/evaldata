"""Tests for `QueryRunner` — budget-aware derived-query execution and scalar read-back."""

import pytest
from pydantic import TypeAdapter, ValidationError

from evaldata.scorers import QueryRunner, ScalarFailure, ScalarResult, ScalarSuccess
from evaldata.types import (
    Column,
    ExecutionError,
    ExecutionFailure,
    ExecutionResult,
    ExecutionSuccess,
    Sql,
    SqlType,
    TypedSchema,
)


def _error(message: str) -> ExecutionError:
    return ExecutionError(kind="query_failed", message=message)


class _RecordingAdapter:
    def __init__(self, results: list[ExecutionResult]) -> None:
        self.executed: list[str] = []
        self._results = list(results)

    def execute(self, sql: str) -> ExecutionResult:
        self.executed.append(sql)
        return self._results.pop(0)

    def cancel(self) -> None: ...

    def close(self) -> None: ...


class _UnresolvedAdapter(_RecordingAdapter):
    """A fake backend that leaves parameters unresolved: its type probe runs through the runner's budgeted `execute`."""

    def type_probe_sql(self, sql: str) -> str:
        return f"PROBE {sql}"

    def types_from_probe(self, rows: list[dict[str, object]]) -> list[SqlType] | ExecutionError:
        if not rows:
            return ExecutionError(kind="type_probe_failed", message="probe returned no rows")
        return [SqlType.parse(str(r["type"]), "databricks") for r in rows]


def _schema(*cols: tuple[str, str]) -> TypedSchema:
    return TypedSchema(root=[Column(name=n, type=SqlType.parse(t, "databricks"), nullable=None) for n, t in cols])


def _runner(results: list[ExecutionResult], budget: float | None) -> tuple[QueryRunner, _RecordingAdapter]:
    adapter = _RecordingAdapter(results)
    return QueryRunner(adapter, Sql("SELECT 1"), "duckdb", budget), adapter


@pytest.mark.unit
class TestQueryRunner:
    def test_success_passes_result_through(self) -> None:
        result = ExecutionSuccess(rows=[{"n": 1}], latency_seconds=0.5)
        runner, adapter = _runner([result], None)
        out = runner.run(Sql("SELECT 1"))
        assert out is result
        assert adapter.executed == ["SELECT 1"]

    def test_model_sql_and_dialect_exposed(self) -> None:
        runner, _ = _runner([], None)
        assert runner.model_sql == "SELECT 1"
        assert runner.dialect == "duckdb"

    def test_underlying_error_returned_after_executing(self) -> None:
        result = ExecutionFailure(latency_seconds=0.0, error=_error("boom"))
        runner, adapter = _runner([result], None)
        out = runner.run(Sql("SELECT bad"))
        assert isinstance(out, ExecutionFailure)
        assert out.error.message == "boom"
        assert adapter.executed == ["SELECT bad"]

    def test_pool_decrements_across_calls(self) -> None:
        results = [
            ExecutionSuccess(rows=[], latency_seconds=2.0),
            ExecutionSuccess(rows=[], latency_seconds=2.0),
        ]
        runner, adapter = _runner(results, 5.0)
        runner.run(Sql("SELECT 1"))
        runner.run(Sql("SELECT 2"))
        assert len(adapter.executed) == 2

    def test_exhausted_pool_fails_fast(self) -> None:
        results = [
            ExecutionSuccess(rows=[], latency_seconds=1.0),
            ExecutionSuccess(rows=[], latency_seconds=1.0),
        ]
        runner, adapter = _runner(results, 1.0)
        runner.run(Sql("SELECT 1"))
        out = runner.run(Sql("SELECT 2"))
        assert isinstance(out, ExecutionFailure)
        assert out.error.kind == "budget_exceeded"
        assert "budget" in out.error.message
        assert len(adapter.executed) == 1

    def test_unbounded_pool_never_short_circuits(self) -> None:
        results = [
            ExecutionSuccess(rows=[], latency_seconds=10.0),
            ExecutionSuccess(rows=[], latency_seconds=10.0),
        ]
        runner, adapter = _runner(results, None)
        runner.run(Sql("SELECT 1"))
        runner.run(Sql("SELECT 2"))
        assert len(adapter.executed) == 2


@pytest.mark.unit
class TestScalar:
    def test_single_cell_success(self) -> None:
        runner, adapter = _runner([ExecutionSuccess(rows=[{"n": 3}], latency_seconds=0.2)], None)
        out = runner.scalar(Sql("SELECT count(*)"))
        assert isinstance(out, ScalarSuccess)
        assert out.value == 3
        assert out.latency_seconds == 0.2
        assert adapter.executed == ["SELECT count(*)"]

    def test_sql_null_is_a_successful_scalar(self) -> None:
        runner, _ = _runner([ExecutionSuccess(rows=[{"value": None}], latency_seconds=0.1)], None)
        out = runner.scalar(Sql("SELECT NULL"))
        assert isinstance(out, ScalarSuccess)
        assert out.value is None

    def test_success_round_trips_with_sql_null(self) -> None:
        out = ScalarSuccess(value=None, latency_seconds=0.1)
        restored = TypeAdapter(ScalarResult).validate_json(out.model_dump_json())
        assert restored == out

    def test_underlying_error_propagated(self) -> None:
        runner, _ = _runner([ExecutionFailure(latency_seconds=0.1, error=_error("boom"))], None)
        out = runner.scalar(Sql("SELECT bad"))
        assert isinstance(out, ScalarFailure)
        assert out.error.message == "boom"
        assert out.latency_seconds == 0.1

    def test_failure_round_trips(self) -> None:
        out = ScalarFailure(error=_error("boom"), latency_seconds=0.1)
        restored = TypeAdapter(ScalarResult).validate_json(out.model_dump_json())
        assert restored == out

    def test_success_rejects_error(self) -> None:
        with pytest.raises(ValidationError, match="error"):
            ScalarSuccess(value=1, error=_error("boom"), latency_seconds=0.1)  # type: ignore[call-arg]

    def test_failure_rejects_value(self) -> None:
        with pytest.raises(ValidationError, match="value"):
            ScalarFailure(value=None, error=_error("boom"), latency_seconds=0.1)  # type: ignore[call-arg]

    def test_zero_rows_is_error(self) -> None:
        runner, _ = _runner([ExecutionSuccess(rows=[], latency_seconds=0.0)], None)
        out = runner.scalar(Sql("SELECT 1 WHERE 1=0"))
        assert isinstance(out, ScalarFailure)
        assert out.error.message == "expected one row and one column"

    def test_multiple_rows_is_error(self) -> None:
        runner, _ = _runner([ExecutionSuccess(rows=[{"n": 1}, {"n": 2}], latency_seconds=0.0)], None)
        out = runner.scalar(Sql("SELECT n"))
        assert isinstance(out, ScalarFailure)
        assert out.error.message == "expected one row and one column"

    def test_multiple_columns_is_error(self) -> None:
        runner, _ = _runner([ExecutionSuccess(rows=[{"a": 1, "b": 2}], latency_seconds=0.0)], None)
        out = runner.scalar(Sql("SELECT a, b"))
        assert isinstance(out, ScalarFailure)
        assert out.error.message == "expected one row and one column"

    def test_draws_down_budget_pool(self) -> None:
        results = [
            ExecutionSuccess(rows=[{"n": 1}], latency_seconds=1.0),
            ExecutionSuccess(rows=[{"n": 1}], latency_seconds=1.0),
        ]
        runner, adapter = _runner(results, 1.0)
        runner.scalar(Sql("SELECT 1"))
        out = runner.scalar(Sql("SELECT 2"))
        assert isinstance(out, ScalarFailure)
        assert "budget" in out.error.message
        assert len(adapter.executed) == 1


@pytest.mark.unit
class TestResolvedSchema:
    def test_precise_backend_returns_base_unchanged(self) -> None:
        base = _schema(("amount", "decimal"))
        runner, _ = _runner([], None)  # _RecordingAdapter is not a TypeResolvingAdapter
        assert runner.resolved_schema(base, Sql("SELECT amount")) is base

    def test_unresolved_backend_probes_through_runner_and_grafts_by_position(self) -> None:
        probe = ExecutionSuccess(
            rows=[{"type": "decimal(10,2)"}, {"type": "array<string>"}],
            latency_seconds=0.1,
        )
        adapter = _UnresolvedAdapter([probe])
        runner = QueryRunner(adapter, Sql("SELECT amount, tags"), "databricks", None)
        out = runner.resolved_schema(_schema(("amount", "decimal"), ("tags", "array")), runner.model_sql)
        assert isinstance(out, TypedSchema)
        assert out.types == [
            SqlType.parse("decimal(10,2)", "databricks"),
            SqlType.parse("array<string>", "databricks"),
        ]
        assert out.names == ["amount", "tags"]  # names/order kept from base
        assert adapter.executed == ["PROBE SELECT amount, tags"]  # probe ran through the runner

    def test_column_count_mismatch_is_error(self) -> None:
        # Probe yields fewer columns than the result has → surfaced, never silently unresolved.
        probe = ExecutionSuccess(rows=[{"type": "decimal(10,2)"}], latency_seconds=0.0)
        adapter = _UnresolvedAdapter([probe])
        runner = QueryRunner(adapter, Sql("SELECT amount, note"), "databricks", None)
        out = runner.resolved_schema(_schema(("amount", "decimal"), ("note", "string")), runner.model_sql)
        assert isinstance(out, ExecutionError)
        assert "1 column type(s) for a 2-column result" in out.message

    def test_probe_query_error_propagates(self) -> None:
        adapter = _UnresolvedAdapter([ExecutionFailure(latency_seconds=0.0, error=_error("warehouse down"))])
        runner = QueryRunner(adapter, Sql("SELECT n"), "databricks", None)
        out = runner.resolved_schema(_schema(("n", "int")), runner.model_sql)
        assert isinstance(out, ExecutionError)
        assert out.message == "warehouse down"

    def test_probe_parse_error_propagates(self) -> None:
        adapter = _UnresolvedAdapter([ExecutionSuccess(rows=[], latency_seconds=0.0)])  # success, no rows
        runner = QueryRunner(adapter, Sql("SELECT n"), "databricks", None)
        out = runner.resolved_schema(_schema(("n", "int")), runner.model_sql)
        assert isinstance(out, ExecutionError)
        assert out.message == "probe returned no rows"

    def test_probe_respects_exhausted_budget(self) -> None:
        adapter = _UnresolvedAdapter([ExecutionSuccess(rows=[{"name": "n", "type": "int"}], latency_seconds=0.0)])
        runner = QueryRunner(adapter, Sql("SELECT n"), "databricks", 0.0)  # pool already exhausted
        out = runner.resolved_schema(_schema(("n", "int")), runner.model_sql)
        assert isinstance(out, ExecutionError)
        assert "budget" in out.message
        assert adapter.executed == []  # the probe never reached the adapter
