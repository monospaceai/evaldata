"""Tests for `DatabricksAdapter`."""

from typing import Any

import pytest

from evaldata.types import ExecutionFailure, ExecutionSuccess

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
        self.open = True

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


def _adapter(cursor: _FakeCursor, *, active: bool = False) -> DatabricksAdapter:
    adapter = object.__new__(DatabricksAdapter)
    adapter._conn = _FakeConnection(cursor)
    adapter._cursor = cursor if active else None
    return adapter


@pytest.mark.unit
class TestExecute:
    def test_rows_and_schema_on_success(self) -> None:
        cursor = _FakeCursor([("n", "int"), ("amount", "decimal")], [(1, "9.50"), (2, "8.00")])
        result = _adapter(cursor).execute("SELECT n, amount FROM t")
        assert isinstance(result, ExecutionSuccess)
        assert result.rows == [{"n": 1, "amount": "9.50"}, {"n": 2, "amount": "8.00"}]
        assert result.schema_ is not None
        assert result.schema_.names == ["n", "amount"]
        assert all(c.nullable is None for c in result.schema_.root)

    def test_error_is_returned_not_raised(self) -> None:
        result = _adapter(_FakeCursor(None, [], error="boom")).execute("SELECT bad")
        assert isinstance(result, ExecutionFailure)
        assert result.error.message == "boom"

    def test_non_row_returning_statement_has_no_schema(self) -> None:
        result = _adapter(_FakeCursor(None, [])).execute("CREATE TABLE t (n INT)")
        assert isinstance(result, ExecutionSuccess)
        assert result.rows == []
        assert result.schema_ is None

    def test_duplicate_names_error_before_fetch(self) -> None:
        class _NoFetchCursor(_FakeCursor):
            def fetchall(self) -> list[tuple[Any, ...]]:
                msg = "fetchall must not run when output names are duplicated"
                raise AssertionError(msg)

        result = _adapter(_NoFetchCursor([("x", "int"), ("x", "int")], [])).execute("SELECT 1 AS x, 2 AS x")
        assert isinstance(result, ExecutionFailure)
        assert "duplicate output column name(s)" in result.error.message

    def test_non_connector_fetch_error_is_returned_not_raised(self) -> None:
        class _ArrowFailCursor(_FakeCursor):
            def fetchall(self) -> list[tuple[Any, ...]]:
                msg = "Can't unify schema with duplicate field names"
                raise ValueError(msg)

        result = _adapter(_ArrowFailCursor([("n", "int")], [])).execute("SELECT n")
        assert isinstance(result, ExecutionFailure)
        assert result.error.message == "Can't unify schema with duplicate field names"


@pytest.mark.unit
class TestTypeProbe:
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
        assert callable(captured["credentials_provider"])

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

    def test_ping_and_disconnect_classification_distinguish_server_errors(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        adapter = _adapter(_FakeCursor(None, []))
        assert adapter.ping() is True
        adapter._conn.open = False  # noqa: SLF001
        assert adapter.ping() is False
        from databricks.sql import exc as databricks_exc

        closed = ExecutionError(kind="query_failed", message="x", cause=databricks_exc.SessionAlreadyClosedError("x"))
        ordinary = ExecutionError(kind="query_failed", message="x", cause=databricks_exc.ServerOperationError("x"))
        sqlstate = ExecutionError(kind="query_failed", message="x", sqlstate="08006")
        assert adapter.is_disconnect(sqlstate) is True
        assert adapter.is_disconnect(closed) is True
        assert adapter.is_disconnect(ordinary) is False
        monkeypatch.setattr(adapter, "ping", lambda: pytest.fail("disconnect classification must not perform I/O"))
        interface = ExecutionError(kind="query_failed", message="x", cause=databricks_exc.InterfaceError("x"))
        assert adapter.is_disconnect(interface) is True

    def test_ping_treats_cursor_failure_as_unhealthy(self) -> None:
        adapter = _adapter(_FakeCursor(None, [], error="unavailable"))
        assert adapter.ping() is False

    def test_cancel_is_a_noop_when_no_query_runs(self) -> None:
        _adapter(_FakeCursor(None, [])).cancel()

    def test_cancel_swallows_errors(self) -> None:
        class _BadCancelCursor(_FakeCursor):
            def cancel(self) -> None:
                msg = "cancel failed"
                raise RuntimeError(msg)

        _adapter(_BadCancelCursor(None, []), active=True).cancel()

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
@pytest.mark.databricks
class TestTypeResolutionLive:
    def test_decimal_and_array_types_resolve_to_precise(self) -> None:
        from .conftest import connect_databricks

        adapter = connect_databricks()
        try:
            sql = Sql("SELECT CAST(1.5 AS DECIMAL(10,2)) AS amount, ARRAY('a', 'b') AS tags")
            result = adapter.execute(sql)
            assert isinstance(result, ExecutionSuccess)
            assert result.schema_ is not None

            resolved = QueryRunner(adapter, sql, "databricks", None).resolved_schema(result.schema_, sql)
            assert not isinstance(resolved, ExecutionError), f"type resolution failed: {resolved}"
            precise = {c.name: c.type for c in resolved.root}
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
            assert isinstance(result, ExecutionSuccess)
            assert result.schema_ is not None

            resolved = QueryRunner(adapter, sql, "databricks", None).resolved_schema(result.schema_, sql)
            assert not isinstance(resolved, ExecutionError), f"type resolution failed: {resolved}"
            precise = {c.name: c.type for c in resolved.root}
            assert precise["amount"] == SqlType.parse("decimal(10,2)", "databricks"), precise["amount"].raw
            assert precise["s"] == SqlType.parse("struct<x:int,y:string>", "databricks"), precise["s"].raw
        finally:
            adapter.close()
