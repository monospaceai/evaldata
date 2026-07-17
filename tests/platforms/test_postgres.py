"""Postgres-specific tests: native-type-string fidelity via psycopg `type_display`."""

import pytest
from sqlglot import exp

from evaldata import CallableSolver, EvalCase, ExecutionAccuracy, run_benchmark
from evaldata.platforms import postgres_platform
from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.registry import close_all, resolve
from evaldata.types import GoldQuery

from .conftest import _postgres_dsn, connect_postgres


@pytest.mark.e2e
class TestPostgresNativeTypes:
    """Postgres emits the native type strings (psycopg `type_display`) SQLGlot's `postgres` dialect parses."""

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
        assert result.error is None
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
        """The type strings PostgresAdapter emits must round-trip through SQLGlot."""
        result = adapter.execute(sql)
        assert result.error is None
        assert result.schema_ is not None
        parsed = exp.DataType.build(result.schema_[0].type.raw, dialect="postgres")
        assert isinstance(parsed, exp.DataType)


@pytest.mark.e2e
class TestPostgresLifecycle:
    """Connection lifecycle and non-row-returning statements."""

    def test_context_manager_returns_self_and_closes(self) -> None:
        connect_postgres().close()  # fail early if Postgres is unreachable
        from evaldata.platforms.postgres import PostgresAdapter

        from .conftest import _postgres_dsn

        with PostgresAdapter(_postgres_dsn()) as adapter:
            assert adapter.execute("SELECT 1 AS n").error is None
        # Exit closes the connection; a later execute now fails as a value rather than succeeding.
        assert adapter.execute("SELECT 1 AS n").error is not None

    def test_non_row_returning_statement_succeeds_without_schema(self) -> None:
        adapter = connect_postgres()
        try:
            result = adapter.execute("CREATE TEMP TABLE t_cov (x int)")
            assert result.error is None
            assert result.schema_ is None
            assert result.rows == []
        finally:
            adapter.close()


@pytest.mark.e2e
class TestPostgresConcurrency:
    """Dedicated-connection pool: concurrent cases each get their own connection and stay correct."""

    def test_concurrent_cases_score_independently(self) -> None:
        connect_postgres().close()  # fail early if Postgres is unreachable
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
