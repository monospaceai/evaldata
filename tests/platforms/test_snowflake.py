"""Tests for `SnowflakeAdapter`."""

from typing import Any

import pytest

snowflake_connector = pytest.importorskip("snowflake.connector")

from snowflake.connector.constants import FIELD_NAME_TO_ID  # noqa: E402
from snowflake.connector.cursor import ResultMetadata  # noqa: E402

from evaldata.platforms.snowflake import SnowflakeAdapter, _type_string  # noqa: E402
from evaldata.types import SqlType  # noqa: E402


def _meta(
    name: str,
    field: str,
    *,
    precision: int | None = None,
    scale: int | None = None,
    internal_size: int | None = None,
    is_nullable: bool = True,
) -> ResultMetadata:
    return ResultMetadata(
        name=name,
        type_code=FIELD_NAME_TO_ID[field],
        display_size=None,
        internal_size=internal_size,
        precision=precision,
        scale=scale,
        is_nullable=is_nullable,
    )


class _FakeCursor:
    def __init__(
        self,
        description: list[ResultMetadata] | None,
        rows: list[tuple[Any, ...]],
        error: str | None = None,
        sfqid: str | None = "01a",
    ) -> None:
        self._description = description
        self._rows = rows
        self._error = error
        self.sfqid = sfqid
        self.executed: str | None = None
        self.aborted: str | None = None
        self.closed = False

    def execute(self, sql: str, timeout: int | None = None) -> None:
        self.executed = sql
        self.timeout = timeout
        if self._error is not None:
            raise snowflake_connector.errors.ProgrammingError(self._error)

    @property
    def description(self) -> list[ResultMetadata] | None:
        return self._description

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def abort_query(self, qid: str) -> None:
        self.aborted = qid

    def close(self) -> None:
        self.closed = True


class _FakeConnection:
    def __init__(self, cursor: _FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False
        self.valid = True

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True

    def is_valid(self) -> bool:
        return self.valid


def _adapter(cursor: _FakeCursor, *, active: bool = False) -> SnowflakeAdapter:
    adapter = object.__new__(SnowflakeAdapter)
    adapter._conn = _FakeConnection(cursor)
    adapter._cursor = cursor if active else None
    return adapter


@pytest.mark.unit
class TestExecute:
    def test_native_timeout_rounds_up_and_executes_sql_once(self) -> None:
        cursor = _FakeCursor(None, [])
        result = _adapter(cursor).execute_with_timeout("SELECT 1", 1.01)
        assert result.error is None
        assert cursor.executed == "SELECT 1"
        assert cursor.timeout == 2

    def test_rows_and_schema_on_success(self) -> None:
        description = [
            _meta("id", "FIXED", precision=10, scale=2),
            _meta("label", "TEXT", internal_size=16777216),
        ]
        cursor = _FakeCursor(description, [(1, "a"), (2, "b")])
        result = _adapter(cursor).execute("SELECT id, label FROM t")
        assert result.error is None
        assert result.rows == [{"id": 1, "label": "a"}, {"id": 2, "label": "b"}]
        assert result.schema_ is not None
        assert result.schema_.names == ["id", "label"]
        columns = {c.name: c for c in result.schema_.root}
        assert columns["id"].type == SqlType.parse("NUMBER(10,2)", "snowflake")
        assert columns["label"].type == SqlType.parse("VARCHAR(16777216)", "snowflake")

    def test_nullable_comes_through_from_metadata(self) -> None:
        description = [_meta("id", "FIXED", is_nullable=False), _meta("x", "TEXT", is_nullable=True)]
        result = _adapter(_FakeCursor(description, [(1, "a")])).execute("SELECT id, x FROM t")
        assert result.schema_ is not None
        columns = {c.name: c for c in result.schema_.root}
        assert columns["id"].nullable is False
        assert columns["x"].nullable is True

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

    def test_duplicate_names_error(self) -> None:
        description = [_meta("x", "FIXED", precision=38, scale=0), _meta("x", "FIXED", precision=38, scale=0)]
        result = _adapter(_FakeCursor(description, [(1, 2)])).execute("SELECT 1 AS x, 2 AS x")
        assert result.rows == []
        assert result.schema_ is None
        assert result.error is not None
        assert "duplicate output column name(s)" in result.error.message


@pytest.mark.unit
class TestTypeString:
    def test_fixed_with_precision(self) -> None:
        assert _type_string("FIXED", 38, 0, None) == "NUMBER(38,0)"

    def test_fixed_without_precision(self) -> None:
        assert _type_string("FIXED", None, None, None) == "NUMBER"

    def test_real(self) -> None:
        assert _type_string("REAL", None, None, None) == "FLOAT"

    def test_text_with_internal_size(self) -> None:
        assert _type_string("TEXT", None, None, 16777216) == "VARCHAR(16777216)"

    def test_text_without_internal_size(self) -> None:
        assert _type_string("TEXT", None, None, None) == "VARCHAR"

    def test_binary_with_internal_size(self) -> None:
        assert _type_string("BINARY", None, None, 8388608) == "BINARY(8388608)"

    def test_binary_without_internal_size(self) -> None:
        assert _type_string("BINARY", None, None, None) == "BINARY"

    def test_time_with_scale(self) -> None:
        assert _type_string("TIME", None, 9, None) == "TIME(9)"

    def test_time_without_scale(self) -> None:
        assert _type_string("TIME", None, None, None) == "TIME"

    @pytest.mark.parametrize("field", ["TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ"])
    def test_timestamp_variants_with_scale(self, field: str) -> None:
        assert _type_string(field, None, 9, None) == f"{field}(9)"

    @pytest.mark.parametrize("field", ["TIMESTAMP_NTZ", "TIMESTAMP_LTZ", "TIMESTAMP_TZ"])
    def test_timestamp_variants_without_scale(self, field: str) -> None:
        assert _type_string(field, None, None, None) == field

    def test_timestamp(self) -> None:
        assert _type_string("TIMESTAMP", None, None, None) == "TIMESTAMP"

    @pytest.mark.parametrize("field", ["BOOLEAN", "DATE", "VARIANT", "OBJECT", "ARRAY", "GEOGRAPHY", "GEOMETRY"])
    def test_passthrough_types(self, field: str) -> None:
        assert _type_string(field, None, None, None) == field

    def test_unknown_field_name_returned_unchanged(self) -> None:
        assert _type_string("VECTOR", None, None, None) == "VECTOR"


@pytest.mark.unit
class TestLifecycle:
    @staticmethod
    def _patch_connect(monkeypatch: pytest.MonkeyPatch, captured: dict[str, Any]) -> None:
        def fake_connect(**kwargs: Any) -> _FakeConnection:
            captured.update(kwargs)
            return _FakeConnection(_FakeCursor(None, []))

        monkeypatch.setattr(snowflake_connector, "connect", fake_connect)

    def test_init_connects_with_provided_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_connect(monkeypatch, captured)
        SnowflakeAdapter(account="acc", user="u", password="p", warehouse="wh")
        assert captured["account"] == "acc"
        assert captured["user"] == "u"
        assert captured["password"] == "p"
        assert captured["warehouse"] == "wh"

    def test_init_omits_unset_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_connect(monkeypatch, captured)
        SnowflakeAdapter(account="acc")
        assert captured == {"account": "acc"}
        assert "user" not in captured
        assert "password" not in captured

    def test_init_passes_through_key_pair_kwargs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, Any] = {}
        self._patch_connect(monkeypatch, captured)
        SnowflakeAdapter(account="acc", private_key_file="/k.p8", private_key_file_pwd="pw")
        assert captured["private_key_file"] == "/k.p8"
        assert captured["private_key_file_pwd"] == "pw"

    def test_connection_exposes_the_underlying_connection(self) -> None:
        adapter = _adapter(_FakeCursor(None, []))
        assert adapter.connection is adapter._conn  # noqa: SLF001

    def test_cancel_aborts_the_active_cursor(self) -> None:
        cursor = _FakeCursor(None, [])
        _adapter(cursor, active=True).cancel()
        assert cursor.aborted == "01a"

    def test_ping_and_disconnect_classification_use_connection_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        adapter = _adapter(_FakeCursor(None, []))
        assert adapter.ping() is True
        adapter._conn.valid = False  # noqa: SLF001
        assert adapter.ping() is False
        from evaldata.types import ExecutionError

        assert adapter.is_disconnect(ExecutionError(kind="query_failed", message="x", sqlstate="08006")) is True
        monkeypatch.setattr(adapter, "ping", lambda: pytest.fail("disconnect classification must not perform I/O"))
        ambiguous = snowflake_connector.errors.OperationalError("connection error")
        assert adapter.is_disconnect(ExecutionError(kind="query_failed", message="x", cause=ambiguous)) is True

    def test_ping_treats_connector_failure_as_unhealthy(self) -> None:
        adapter = _adapter(_FakeCursor(None, []))

        def fail() -> bool:
            msg = "is_valid failed"
            raise RuntimeError(msg)

        adapter._conn.is_valid = fail  # type: ignore[method-assign]  # noqa: SLF001
        assert adapter.ping() is False

    def test_cancel_is_a_noop_when_no_query_runs(self) -> None:
        _adapter(_FakeCursor(None, [])).cancel()

    def test_cancel_is_a_noop_when_sfqid_is_falsy(self) -> None:
        cursor = _FakeCursor(None, [], sfqid=None)
        _adapter(cursor, active=True).cancel()
        assert cursor.aborted is None

    def test_cancel_swallows_errors(self) -> None:
        class _BadAbortCursor(_FakeCursor):
            def abort_query(self, qid: str) -> None:
                msg = "abort failed"
                raise RuntimeError(msg)

        _adapter(_BadAbortCursor(None, []), active=True).cancel()

    def test_close_releases_the_connection(self) -> None:
        conn = _FakeConnection(_FakeCursor(None, []))
        adapter = object.__new__(SnowflakeAdapter)
        adapter._conn = conn
        adapter._cursor = None
        adapter.close()
        assert conn.closed is True

    def test_context_manager_returns_self_and_closes(self) -> None:
        conn = _FakeConnection(_FakeCursor(None, []))
        adapter = object.__new__(SnowflakeAdapter)
        adapter._conn = conn
        adapter._cursor = None
        with adapter as entered:
            assert entered is adapter
        assert conn.closed is True


@pytest.mark.e2e
@pytest.mark.cloud
@pytest.mark.snowflake
class TestConcurrencySmoke:
    def test_trivial_selects_run_concurrently(self) -> None:
        import os

        from evaldata import CallableSolver, EvalCase, ExecutionAccuracy, run_benchmark
        from evaldata.platforms import snowflake_platform
        from evaldata.platforms.registry import close_all
        from evaldata.types import GoldQuery

        platform = snowflake_platform(
            name="sf-conc-smoke",
            account=os.environ["SNOWFLAKE_ACCOUNT"],
            user=os.environ.get("SNOWFLAKE_USER"),
            warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
            role=os.environ.get("SNOWFLAKE_ROLE"),
            database=os.environ.get("SNOWFLAKE_DATABASE"),
            schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        )
        try:
            cases = [
                EvalCase(
                    id=f"n-{n}",
                    input=str(n),
                    expected=GoldQuery(sql=f"SELECT {n} AS n"),
                    platform=platform,
                )
                for n in range(4)
            ]
            solver = CallableSolver(lambda c: f"SELECT {c.input} AS n")
            summary = run_benchmark(cases, solver, scorers=[ExecutionAccuracy()], max_concurrency=4)
            assert summary.passed == summary.total == 4
        finally:
            close_all()


@pytest.mark.e2e
@pytest.mark.cloud
@pytest.mark.snowflake
class TestTypeResolutionLive:
    def test_number_and_varchar_types_resolve_to_precise(self) -> None:
        from .conftest import connect_snowflake

        adapter = connect_snowflake()
        try:
            result = adapter.execute("SELECT CAST(1.5 AS NUMBER(10,2)) AS amount, CAST('x' AS VARCHAR(50)) AS label")
            assert result.error is None, result.error
            assert result.schema_ is not None
            precise = {c.name: c.type for c in result.schema_.root}
            assert precise["AMOUNT"] == SqlType.parse("NUMBER(10,2)", "snowflake"), precise["AMOUNT"].raw
            assert precise["LABEL"] == SqlType.parse("VARCHAR(50)", "snowflake"), precise["LABEL"].raw
        finally:
            adapter.close()
