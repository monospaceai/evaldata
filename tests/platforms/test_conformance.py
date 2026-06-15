"""Shared `PlatformAdapter` conformance battery, parametrised over every adapter via `under_test`."""

from dataeval.platforms.base import PlatformAdapter, execute_within_budget

from .conftest import UnderTest


def test_satisfies_platform_adapter_protocol(under_test: UnderTest) -> None:
    assert isinstance(under_test.adapter, PlatformAdapter)


def test_execute_returns_rows_and_schema(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.one_row_one_column)
    assert result.error is None
    assert result.rows == [{"n": 1}]
    assert result.schema_ is not None
    assert len(result.schema_) == 1
    assert result.schema_[0].name == "n"
    assert result.schema_[0].type  # non-empty native type string


def test_empty_result_set_keeps_schema(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.empty_result)
    assert result.error is None
    assert result.rows == []
    assert result.schema_ is not None
    assert len(result.schema_) == 1
    assert result.schema_[0].name == "n"


def test_multiple_rows_returned(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.three_rows)
    assert result.error is None
    assert len(result.rows) == 3
    assert sorted(r["n"] for r in result.rows) == [1, 2, 3]


def test_null_values_round_trip(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.null_value)
    assert result.error is None
    assert result.rows == [{"x": None}]


def test_duplicate_output_columns_return_error(under_test: UnderTest) -> None:
    # Name-keyed rows cannot represent two columns sharing a name; the adapter surfaces
    # this as an error rather than silently dropping the colliding column.
    result = under_test.adapter.execute(under_test.fixtures.duplicate_column_names)
    assert result.error is not None
    assert "duplicate" in result.error
    assert result.rows == []
    assert result.schema_ is None


def test_missing_table_returns_error_not_exception(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.references_missing_table)
    assert result.error is not None
    assert result.error  # non-empty
    assert result.rows == []
    assert result.schema_ is None


def test_parse_error_returns_error_not_exception(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.parse_error)
    assert result.error is not None
    assert result.error  # non-empty
    assert result.rows == []
    assert result.schema_ is None


def test_latency_is_measured_on_success(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.one_row_one_column)
    assert result.error is None
    assert result.latency_seconds >= 0


def test_latency_is_measured_on_failure(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.parse_error)
    assert result.error is not None
    assert result.latency_seconds >= 0


def test_query_within_budget_returns_result(under_test: UnderTest) -> None:
    result = execute_within_budget(under_test.adapter, under_test.fixtures.one_row_one_column, max_seconds=30)
    assert result.error is None
    assert result.rows == [{"n": 1}]


def test_query_exceeding_budget_is_cancelled(under_test: UnderTest) -> None:
    # A query that overruns the budget is aborted via adapter.cancel() and surfaced as an
    # error rather than blocking for its full runtime.
    result = execute_within_budget(under_test.adapter, under_test.fixtures.slow_query, max_seconds=0.5)
    assert result.error is not None
    assert "budget" in result.error
    assert result.rows == []
    assert result.schema_ is None
    assert result.latency_seconds >= 0.5


def test_cancel_is_safe_when_no_query_running(under_test: UnderTest) -> None:
    # cancel() with nothing in flight is a no-op; the adapter stays usable afterwards.
    under_test.adapter.cancel()
    result = under_test.adapter.execute(under_test.fixtures.one_row_one_column)
    assert result.error is None
    assert result.rows == [{"n": 1}]
