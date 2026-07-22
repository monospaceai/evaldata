"""Tests for `PostgresAdapter`."""

import psycopg
import pytest
from sqlglot import exp

from evaldata import CallableSolver, EvalCase, ExecutionAccuracy, run_benchmark
from evaldata.platforms import postgres_platform
from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.postgres import PostgresAdapter
from evaldata.platforms.registry import close_all, resolve
from evaldata.types import ExecutionError, ExecutionFailure, ExecutionSuccess, GoldQuery

from .conftest import _postgres_dsn, connect_postgres


class _TimeoutCursor:
    def __init__(self, *, user_error: Exception | None = None, restore_error: Exception | None = None) -> None:
        self.user_error = user_error
        self.restore_error = restore_error
        self.calls: list[tuple[str, tuple[object, ...] | None]] = []
        self.description = None

    def __enter__(self) -> "_TimeoutCursor":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
        self.calls.append((sql, params))
        if sql == "SELECT user_sql" and self.user_error is not None:
            raise self.user_error
        if sql.startswith("SELECT set_config") and self.restore_error is not None and params == ("1s",):
            raise self.restore_error

    def fetchone(self) -> tuple[str]:
        return ("1s",)

    def fetchall(self) -> list[tuple[object, ...]]:
        return []


class _TimeoutConnection:
    def __init__(self, *cursors: _TimeoutCursor, closed: bool = False, broken: bool = False) -> None:
        self._cursors = list(cursors)
        self.closed = closed
        self.broken = broken
        self.cancelled = False

    def cursor(self) -> _TimeoutCursor:
        return self._cursors.pop(0)

    def cancel_safe(self) -> None:
        self.cancelled = True

    def close(self) -> None:
        self.closed = True


def _timeout_adapter(connection: _TimeoutConnection) -> PostgresAdapter:
    adapter = object.__new__(PostgresAdapter)
    adapter._conn = connection
    adapter._reusable = True
    return adapter


@pytest.mark.e2e
class TestPostgresNativeTypes:
    @pytest.fixture
    def adapter(self) -> PlatformAdapter:
        return connect_postgres()

    @pytest.mark.parametrize(
        ("sql", "expected_type"),
        [
            ("SELECT 1::bigint AS x", "int8"),
            ("SELECT 1::integer AS x", "int4"),
            ("SELECT 1::smallint AS x", "int2"),
            ("SELECT 'a'::text AS x", "text"),
            ("SELECT 'a'::varchar(10) AS x", "varchar(10)"),
            ("SELECT 1.5::numeric(10,2) AS x", "numeric(10,2)"),
            ("SELECT 1.5::double precision AS x", "float8"),
            ("SELECT true AS x", "bool"),
            ("SELECT ARRAY['a', 'b'] AS x", "text[]"),
            ("SELECT '{}'::jsonb AS x", "jsonb"),
            ("SELECT '2020-01-01'::date AS x", "date"),
        ],
    )
    def test_native_type_string_is_exact(self, adapter: PlatformAdapter, sql: str, expected_type: str) -> None:
        result = adapter.execute(sql)
        assert isinstance(result, ExecutionSuccess)
        assert result.schema_ is not None
        assert result.schema_[0].type.raw == expected_type

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1::bigint AS x",
            "SELECT 'a'::varchar(10) AS x",
            "SELECT 1.5::numeric(10,2) AS x",
            "SELECT true AS x",
            "SELECT ARRAY['a', 'b'] AS x",
            "SELECT '{}'::jsonb AS x",
        ],
    )
    def test_emitted_type_parses_under_postgres_dialect(self, adapter: PlatformAdapter, sql: str) -> None:
        result = adapter.execute(sql)
        assert isinstance(result, ExecutionSuccess)
        assert result.schema_ is not None
        parsed = exp.DataType.build(result.schema_[0].type.raw, dialect="postgres")
        assert isinstance(parsed, exp.DataType)


@pytest.mark.e2e
class TestPostgresLifecycle:
    def test_context_manager_returns_self_and_closes(self) -> None:
        connect_postgres().close()
        from evaldata.platforms.postgres import PostgresAdapter

        from .conftest import _postgres_dsn

        with PostgresAdapter(_postgres_dsn()) as adapter:
            assert isinstance(adapter.execute("SELECT 1 AS n"), ExecutionSuccess)
        assert isinstance(adapter.execute("SELECT 1 AS n"), ExecutionFailure)

    def test_non_row_returning_statement_succeeds_without_schema(self) -> None:
        adapter = connect_postgres()
        try:
            result = adapter.execute("CREATE TEMP TABLE t_cov (x int)")
            assert isinstance(result, ExecutionSuccess)
            assert result.schema_ is None
            assert result.rows == []
        finally:
            adapter.close()


@pytest.mark.unit
class TestPostgresTimeoutAndHealth:
    def test_init_cancel_close_and_context_lifecycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        connection = _TimeoutConnection()
        monkeypatch.setattr("evaldata.platforms.postgres.psycopg.connect", lambda *args, **kwargs: connection)
        adapter = PostgresAdapter("dsn")
        adapter.cancel()
        assert connection.cancelled is True
        with adapter as entered:
            assert entered is adapter
        assert connection.closed is True

    def test_execute_without_native_timeout_and_missing_setting_are_structured_results(self) -> None:
        ordinary_cursor = _TimeoutCursor()
        ordinary_cursor.description = []
        ordinary = _timeout_adapter(_TimeoutConnection(ordinary_cursor))
        assert isinstance(ordinary.execute("SELECT user_sql"), ExecutionSuccess)

        class MissingSettingCursor(_TimeoutCursor):
            def fetchone(self) -> None:
                return None

        missing = _timeout_adapter(_TimeoutConnection(MissingSettingCursor()))
        assert isinstance(missing.execute_with_timeout("SELECT user_sql", 1), ExecutionFailure)
        assert missing.is_reusable() is False

    def test_native_timeout_rounds_up_executes_user_sql_once_and_restores(self) -> None:
        work = _TimeoutCursor()
        restore = _TimeoutCursor()
        adapter = _timeout_adapter(_TimeoutConnection(work, restore))
        result = adapter.execute_with_timeout("SELECT user_sql", 0.0011)
        assert isinstance(result, ExecutionSuccess)
        assert work.calls == [
            ("SELECT current_setting('statement_timeout')", None),
            ("SELECT set_config('statement_timeout', %s, false)", ("2",)),
            ("SELECT user_sql", None),
        ]
        assert restore.calls == [("SELECT set_config('statement_timeout', %s, false)", ("1s",))]

    def test_native_cancellation_is_budget_error_but_ordinary_cancellation_is_not(self) -> None:
        cancelled = psycopg.errors.QueryCanceled("cancelled")
        native = _timeout_adapter(_TimeoutConnection(_TimeoutCursor(user_error=cancelled), _TimeoutCursor()))
        ordinary = _timeout_adapter(_TimeoutConnection(_TimeoutCursor(user_error=cancelled)))
        native_result = native.execute_with_timeout("SELECT user_sql", 1)
        ordinary_result = ordinary.execute("SELECT user_sql")
        assert isinstance(native_result, ExecutionFailure)
        assert native_result.error.kind == "budget_exceeded"
        assert isinstance(ordinary_result, ExecutionFailure)
        assert ordinary_result.error.kind == "query_failed"

    def test_restore_failure_marks_successful_session_unreusable(self) -> None:
        adapter = _timeout_adapter(
            _TimeoutConnection(_TimeoutCursor(), _TimeoutCursor(restore_error=psycopg.Error("restore failed")))
        )
        assert isinstance(adapter.execute_with_timeout("SELECT user_sql", 1), ExecutionSuccess)
        assert adapter.is_reusable() is False

    def test_ping_fast_fails_closed_or_broken_connection(self) -> None:
        assert _timeout_adapter(_TimeoutConnection(closed=True)).ping() is False
        assert _timeout_adapter(_TimeoutConnection(broken=True)).ping() is False

    def test_ping_and_disconnect_classification_cover_live_connection_state(self) -> None:
        adapter = _timeout_adapter(_TimeoutConnection(_TimeoutCursor()))
        assert adapter.ping() is True
        assert adapter.is_disconnect(ExecutionError(kind="query_failed", message="x", sqlstate="08006")) is True

    def test_ping_treats_driver_failure_as_unhealthy(self) -> None:
        class FailingPingCursor(_TimeoutCursor):
            def execute(self, sql: str, params: tuple[object, ...] | None = None) -> None:
                if sql == "SELECT 1":
                    message = "ping failed"
                    raise psycopg.Error(message)
                super().execute(sql, params)

        assert _timeout_adapter(_TimeoutConnection(FailingPingCursor())).ping() is False


@pytest.mark.e2e
class TestPostgresConcurrency:
    def test_concurrent_cases_score_independently(self) -> None:
        connect_postgres().close()
        platform = postgres_platform(name="pg-conc-e2e", conninfo=_postgres_dsn())
        utility = resolve(platform)
        utility.execute("DROP TABLE IF EXISTS eval_conc")
        utility.execute("CREATE TABLE eval_conc (id INT, grp INT)")
        utility.execute("INSERT INTO eval_conc SELECT g, g % 4 FROM generate_series(0, 199) AS g")
        try:
            cases = [
                EvalCase(
                    id=f"grp-{g}-{i}",
                    input=str(g),
                    expected=GoldQuery(sql=f"SELECT count(*) AS c FROM eval_conc WHERE grp = {g}"),
                    platform=platform,
                )
                for i in range(3)
                for g in range(4)
            ]
            solver = CallableSolver(lambda c: f"SELECT count(*) AS c FROM eval_conc WHERE grp = {c.input}")
            summary = run_benchmark(cases, solver, scorers=[ExecutionAccuracy()], max_concurrency=4)
            assert summary.passed == summary.total == 12
        finally:
            resolve(platform).execute("DROP TABLE IF EXISTS eval_conc")
            close_all()
