"""Shared `PlatformAdapter` conformance battery, parametrised over every adapter via `under_test`."""

from evaldata.platforms.base import PlatformAdapter, execute_within_budget
from evaldata.types import ExecutionFailure, ExecutionSuccess, TypedSchema, UntypedSchema

from .conftest import UnderTest, conform_name


def test_satisfies_platform_adapter_protocol(under_test: UnderTest) -> None:
    assert isinstance(under_test.adapter, PlatformAdapter)


def test_execute_returns_rows_and_schema(under_test: UnderTest) -> None:
    n = conform_name("n", under_test.dialect)
    result = under_test.adapter.execute(under_test.fixtures.one_row_one_column)
    assert isinstance(result, ExecutionSuccess)
    assert result.rows == [{n: 1}]
    assert result.schema_ is not None
    assert result.schema_.names == [n]
    if under_test.fixtures.reports_types:
        assert isinstance(result.schema_, TypedSchema)
        assert result.schema_[0].type.raw  # non-empty native type string
    else:
        # Engines that report no result-column types produce a names-only UntypedSchema.
        assert isinstance(result.schema_, UntypedSchema)


def test_empty_result_set_keeps_schema(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.empty_result)
    assert isinstance(result, ExecutionSuccess)
    assert result.rows == []
    assert result.schema_ is not None
    assert result.schema_.names == [conform_name("n", under_test.dialect)]


def test_multiple_rows_returned(under_test: UnderTest) -> None:
    n = conform_name("n", under_test.dialect)
    result = under_test.adapter.execute(under_test.fixtures.three_rows)
    assert isinstance(result, ExecutionSuccess)
    assert len(result.rows) == 3
    assert sorted(r[n] for r in result.rows) == [1, 2, 3]


def test_null_values_round_trip(under_test: UnderTest) -> None:
    x = conform_name("x", under_test.dialect)
    result = under_test.adapter.execute(under_test.fixtures.null_value)
    assert isinstance(result, ExecutionSuccess)
    assert result.rows == [{x: None}]


def test_duplicate_output_columns_return_error(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.duplicate_column_names)
    if under_test.fixtures.renames_duplicate_columns:
        # The engine disambiguates the names itself, so no collision reaches the adapter.
        assert isinstance(result, ExecutionSuccess)
        assert result.schema_ is not None
        assert len(result.schema_.names) == 2
        assert len(set(result.schema_.names)) == 2
    else:
        # Name-keyed rows cannot represent two columns sharing a name.
        assert isinstance(result, ExecutionFailure)
        assert "duplicate" in result.error.message


def test_missing_table_returns_error_not_exception(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.references_missing_table)
    assert isinstance(result, ExecutionFailure)
    assert result.error  # non-empty


def test_parse_error_returns_error_not_exception(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.parse_error)
    assert isinstance(result, ExecutionFailure)
    assert result.error  # non-empty


def test_latency_is_measured_on_success(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.one_row_one_column)
    assert isinstance(result, ExecutionSuccess)
    assert result.latency_seconds >= 0


def test_latency_is_measured_on_failure(under_test: UnderTest) -> None:
    result = under_test.adapter.execute(under_test.fixtures.parse_error)
    assert isinstance(result, ExecutionFailure)
    assert result.latency_seconds >= 0


def test_query_within_budget_returns_result(under_test: UnderTest) -> None:
    result = execute_within_budget(under_test.adapter, under_test.fixtures.one_row_one_column, max_seconds=30)
    assert isinstance(result, ExecutionSuccess)
    assert result.rows == [{conform_name("n", under_test.dialect): 1}]


def test_query_exceeding_budget_is_cancelled(under_test: UnderTest) -> None:
    # An overrunning query is surfaced as an error rather than blocking for its full runtime.
    result = execute_within_budget(under_test.adapter, under_test.fixtures.slow_query, max_seconds=0.5)
    assert isinstance(result, ExecutionFailure)
    assert "budget" in result.error.message
    assert result.latency_seconds >= 0.5


def test_cancel_is_safe_when_no_query_running(under_test: UnderTest) -> None:
    # cancel() with nothing in flight is a no-op; the adapter stays usable afterwards.
    under_test.adapter.cancel()
    result = under_test.adapter.execute(under_test.fixtures.one_row_one_column)
    assert isinstance(result, ExecutionSuccess)
    assert result.rows == [{conform_name("n", under_test.dialect): 1}]
