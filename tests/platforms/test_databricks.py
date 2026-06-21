"""`DatabricksAdapter` tests: unit (fake driver) + a live type-resolution e2e check."""

from typing import Any

import pytest

databricks_sql = pytest.importorskip("databricks.sql")

from evaldata.platforms.databricks import DatabricksAdapter  # noqa: E402
from evaldata.scorers.query import QueryRunner  # noqa: E402
from evaldata.types import ExecutionError, Sql, SqlType  # noqa: E402

Description = list[tuple[str, str]]


class _FakeCursor:
    def __init__(self, description: Description | None, rows: list[tuple[Any, ...]], error: str | None = None) -> None:
        self._description = description
        self._rows = rows
        self._error = error
        self.executed: str | None = None
        self.cancelled = False
        self.closed = False

    def execute(self, sql: str) -> None:
        self.executed = sql
        if self._error is not None:
            raise databricks_sql.Error(self._error)

    @property
    def description(self) -> Description | None:
        return self._description

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def cancel(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _adapter(cursor: _FakeCursor, *, active: bool = False) -> DatabricksAdapter:
    """Build an adapter bound to a fake connection, bypassing the real `__init__`.

    With `active=True` the cursor is set as the in-flight one, so `cancel` can reach it.
    """
    adapter = object.__new__(DatabricksAdapter)
    adapter._conn = _FakeConnection(cursor)
    adapter._cursor = cursor if active else None
    return adapter


@pytest.mark.unit
class TestExecute:
    def test_rows_and_schema_on_success(self) -> None:
        cursor = _FakeCursor([("n", "int"), ("amount", "decimal")], [(1, "9.50"), (2, "8.00")])
        result = _adapter(cursor).execute("SELECT n, amount FROM t")
        assert result.error is None
        assert result.rows == [{"n": 1, "amount": "9.50"}, {"n": 2, "amount": "8.00"}]
        assert result.schema_ is not None
        assert result.schema_.names == ["n", "amount"]
        assert all(c.nullable is None for c in result.schema_.root)

    def test_error_is_returned_not_raised(self) -> None:
        result = _adapter(_FakeCursor(None, [], error="boom")).execute("SELECT bad")
        assert result.rows == []
        assert result.schema_ is None
        assert result.error is not None
        assert result.error.message == "boom"

    def test_non_row_returning_statement_has_no_schema(self) -> None:
        result = _adapter(_FakeCursor(None, [])).execute("CREATE TABLE t (n INT)")
        assert result.error is None
        assert result.rows == []
        assert result.schema_ is None

    def test_duplicate_names_error_before_fetch(self) -> None:
        # Duplicate output names are detected from the description, so the Arrow fetch (which
        # would raise) is never attempted, and the uniform duplicate error is surfaced.
        class _NoFetchCursor(_FakeCursor):
            def fetchall(self) -> list[tuple[Any, ...]]:
                msg = "fetchall must not run when output names are duplicated"
                raise AssertionError(msg)

        result = _adapter(_NoFetchCursor([("x", "int"), ("x", "int")], [])).execute("SELECT 1 AS x, 2 AS x")
        assert result.rows == []
        assert result.schema_ is None
        assert result.error is not None
        assert "duplicate output column name(s)" in result.error.message

    def test_non_connector_fetch_error_is_returned_not_raised(self) -> None:
        # A pyarrow-layer error during fetch must be caught and returned, not propagated.
        class _ArrowFailCursor(_FakeCursor):
            def fetchall(self) -> list[tuple[Any, ...]]:
                msg = "Can't unify schema with duplicate field names"
                raise ValueError(msg)

        result = _adapter(_ArrowFailCursor([("n", "int")], [])).execute("SELECT n")
        assert result.error is not None
        assert result.error.message == "Can't unify schema with duplicate field names"
        assert result.rows == []
        assert result.schema_ is None


@pytest.mark.unit
class TestTypeProbe:
    """The probe is two pure methods; the runner (not the adapter) executes it."""

    def test_probe_sql_strips_trailing_semicolon_and_whitespace(self) -> None:
        assert DatabricksAdapter.type_probe_sql("SELECT 1 ; ") == "DESCRIBE QUERY SELECT 1"

    def test_types_from_probe_parses_rows_in_order(self) -> None:
        rows = [
            {"col_name": "amount", "data_type": "decimal(10,2)", "comment": ""},
            {"col_name": "tags", "data_type": "array<string>", "comment": ""},
        ]
        assert DatabricksAdapter.types_from_probe(rows) == [
            SqlType.parse("decimal(10,2)", "databricks"),
            SqlType.parse("array<string>", "databricks"),
        ]

    def test_types_from_probe_empty_is_error(self) -> None:
        result = DatabricksAdapter.types_from_probe([])
        assert isinstance(result, ExecutionError)
        assert result.message == "DESCRIBE QUERY returned no rows"

    def test_types_from_probe_missing_data_type_is_error(self) -> None:
        result = DatabricksAdapter.types_from_probe([{"col_name": "amount", "comment": ""}])
        assert isinstance(result, ExecutionError)
        assert result.message.startswith("DESCRIBE QUERY row missing data_type:")


@pytest.mark.unit
class TestLifecycle:
    """Connection lifecycle against a mocked connector — keeps coverage independent of creds."""

    @staticmethod
    def _patch_connect(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
        def fake_connect(**kwargs: Any) -> _FakeConnection:
            captured.update(kwargs)
            return _FakeConnection(_FakeCursor(None, []))

        monkeypatch.setattr(databricks_sql, "connect", fake_connect)
        monkeypatch.setattr("evaldata.platforms.databricks.Config", lambda host: object())

    def test_init_connects_and_resolves_credentials_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_connect(monkeypatch, captured)
        DatabricksAdapter(server_hostname="h", http_path="/p", catalog="main", schema="sales")
        assert captured["server_hostname"] == "h"
        assert captured["http_path"] == "/p"
        assert captured["catalog"] == "main"
        assert captured["schema"] == "sales"
        assert callable(captured["credentials_provider"])  # creds resolve lazily, from the env

    def test_init_omits_unset_catalog_and_schema(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_connect(monkeypatch, captured)
        DatabricksAdapter(server_hostname="h", http_path="/p")
        assert "catalog" not in captured
        assert "schema" not in captured

    def test_cancel_aborts_the_active_cursor(self) -> None:
        cursor = _FakeCursor(None, [])
        _adapter(cursor, active=True).cancel()
        assert cursor.cancelled is True

    def test_cancel_is_a_noop_when_no_query_runs(self) -> None:
        _adapter(_FakeCursor(None, [])).cancel()  # _cursor is None; must not raise

    def test_cancel_swallows_errors(self) -> None:
        class _BadCancelCursor(_FakeCursor):
            def cancel(self) -> None:
                msg = "cancel failed"
                raise RuntimeError(msg)

        _adapter(_BadCancelCursor(None, []), active=True).cancel()  # best-effort: swallowed

    def test_close_releases_the_connection(self) -> None:
        conn = _FakeConnection(_FakeCursor(None, []))
        adapter = object.__new__(DatabricksAdapter)
        adapter._conn = conn
        adapter._cursor = None
        adapter.close()
        assert conn.closed is True

    def test_context_manager_returns_self_and_closes(self) -> None:
        conn = _FakeConnection(_FakeCursor(None, []))
        adapter = object.__new__(DatabricksAdapter)
        adapter._conn = conn
        adapter._cursor = None
        with adapter as entered:
            assert entered is adapter
        assert conn.closed is True


@pytest.mark.e2e
@pytest.mark.cloud
class TestTypeResolutionLive:
    """Live `DESCRIBE QUERY` resolution against a real workspace; unit tests use fakes.

    Fails loud (no skip) when `DATABRICKS_SERVER_HOSTNAME`/`DATABRICKS_HTTP_PATH` are unset.
    """

    def test_decimal_and_array_types_resolve_to_precise(self) -> None:
        from .conftest import connect_databricks

        adapter = connect_databricks()
        try:
            sql = Sql("SELECT CAST(1.5 AS DECIMAL(10,2)) AS amount, ARRAY('a', 'b') AS tags")
            result = adapter.execute(sql)
            assert result.error is None, result.error
            assert result.schema_ is not None

            resolved = QueryRunner(adapter, sql, "databricks", None).resolved_schema(result.schema_, sql)
            assert not isinstance(resolved, ExecutionError), f"type resolution failed: {resolved}"
            precise = {c.name: c.type for c in resolved.root}
            # `.raw` surfaces the real DESCRIBE QUERY string if the assumption is wrong.
            assert precise["amount"] == SqlType.parse("decimal(10,2)", "databricks"), precise["amount"].raw
            assert precise["tags"] == SqlType.parse("array<string>", "databricks"), precise["tags"].raw
        finally:
            adapter.close()

    def test_cte_and_struct_resolve_to_precise(self) -> None:
        from .conftest import connect_databricks

        adapter = connect_databricks()
        try:
            sql = Sql(
                "WITH t AS (SELECT CAST(1.5 AS DECIMAL(10,2)) AS amount, "
                "named_struct('x', 1, 'y', CAST('a' AS STRING)) AS s) SELECT amount, s FROM t"
            )
            result = adapter.execute(sql)
            assert result.error is None, result.error
            assert result.schema_ is not None

            resolved = QueryRunner(adapter, sql, "databricks", None).resolved_schema(result.schema_, sql)
            assert not isinstance(resolved, ExecutionError), f"type resolution failed: {resolved}"
            precise = {c.name: c.type for c in resolved.root}
            assert precise["amount"] == SqlType.parse("decimal(10,2)", "databricks"), precise["amount"].raw
            assert precise["s"] == SqlType.parse("struct<x:int,y:string>", "databricks"), precise["s"].raw
        finally:
            adapter.close()
