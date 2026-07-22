"""Unit tests for platform-ref builders and `PlatformRef` -> adapter resolution."""

import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from evaldata.platforms import (
    bigquery_platform,
    databricks_platform,
    duckdb_platform,
    postgres_platform,
    resolve,
    snowflake_platform,
    sqlite_platform,
)
from evaldata.platforms.registry import acquired, close_all, pool_for
from evaldata.types import (
    BigQueryConfig,
    BigQueryPlatformRef,
    DatabricksConfig,
    DatabricksPlatformRef,
    DuckDBConfig,
    DuckDBPlatformRef,
    ExecutionSuccess,
    PlatformRef,
    PoolPolicy,
    PostgreSQLConfig,
    PostgreSQLPlatformRef,
    SnowflakeConfig,
    SnowflakePlatformRef,
)


@pytest.mark.unit
class TestRefBuilders:
    def test_duckdb_platform_builds_ref(self) -> None:
        ref = duckdb_platform(name="local", path="/tmp/x.duckdb")
        assert ref == DuckDBPlatformRef(name="local", config=DuckDBConfig(path="/tmp/x.duckdb"))

    def test_duckdb_platform_defaults_to_in_memory(self) -> None:
        assert duckdb_platform(name="local").config == DuckDBConfig(path=":memory:")

    def test_postgres_platform_builds_ref(self) -> None:
        ref = postgres_platform(name="warehouse", conninfo="host=db")
        assert ref == PostgreSQLPlatformRef(name="warehouse", config=PostgreSQLConfig(conninfo="host=db"))

    def test_databricks_platform_builds_ref(self) -> None:
        ref = databricks_platform(name="wh", server_hostname="h.databricks.com", http_path="/sql/1.0/warehouses/abc")
        assert ref == DatabricksPlatformRef(
            name="wh",
            dialect="databricks",
            config=DatabricksConfig(server_hostname="h.databricks.com", http_path="/sql/1.0/warehouses/abc"),
        )

    def test_databricks_platform_includes_catalog_and_schema_when_set(self) -> None:
        ref = databricks_platform(name="wh", server_hostname="h", http_path="/p", catalog="main", schema="sales")
        assert ref.config == DatabricksConfig(server_hostname="h", http_path="/p", catalog="main", schema="sales")

    def test_snowflake_platform_builds_ref(self) -> None:
        ref = snowflake_platform(
            name="sf",
            account="acme-test",
            warehouse="COMPUTE_WH",
            role="EVALDATA",
            authenticator="WORKLOAD_IDENTITY",
            workload_identity_provider="OIDC",
        )
        assert ref == SnowflakePlatformRef(
            name="sf",
            dialect="snowflake",
            config=SnowflakeConfig(
                account="acme-test",
                warehouse="COMPUTE_WH",
                role="EVALDATA",
                authenticator="WORKLOAD_IDENTITY",
                workload_identity_provider="OIDC",
            ),
        )

    def test_bigquery_platform_builds_ref(self) -> None:
        ref = bigquery_platform(name="bq", project="my-proj")
        assert ref == BigQueryPlatformRef(name="bq", dialect="bigquery", config=BigQueryConfig(project="my-proj"))

    def test_bigquery_platform_includes_dataset_and_location_when_set(self) -> None:
        ref = bigquery_platform(name="bq", project="my-proj", dataset="analytics", location="EU")
        assert ref.config == BigQueryConfig(project="my-proj", dataset="analytics", location="EU")

    def test_default_pool_is_omitted_from_serialization(self) -> None:
        ref = duckdb_platform(name="local")
        assert "pool" not in ref.model_dump()
        assert '"pool"' not in ref.model_dump_json()


@pytest.mark.unit
class TestResolve:
    def test_resolves_and_executes_duckdb(self, tmp_path: Path) -> None:
        db = tmp_path / "t.duckdb"
        adapter = resolve(duckdb_platform(name="local", path=str(db)))
        adapter.execute("CREATE TABLE t (n INTEGER)")
        adapter.execute("INSERT INTO t VALUES (1), (2)")
        result = adapter.execute("SELECT count(*) AS c FROM t")
        assert isinstance(result, ExecutionSuccess)
        assert result.rows == [{"c": 2}]

    def test_same_name_returns_cached_adapter(self) -> None:
        ref = duckdb_platform(name="local")
        assert resolve(ref) is resolve(ref)

    def test_same_name_different_config_raises(self) -> None:
        resolve(duckdb_platform(name="local", path=":memory:"))
        with pytest.raises(ValueError, match="already bound to a different configuration"):
            resolve(duckdb_platform(name="local", path="/tmp/other.duckdb"))

    def test_unsupported_kind_blocked_before_resolution(self) -> None:
        with pytest.raises(ValidationError):
            TypeAdapter(PlatformRef).validate_python({"name": "wh", "kind": "mysql"})

    def test_close_all_is_idempotent_when_empty(self) -> None:
        close_all()
        close_all()

    def test_postgres_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "evaldata.platforms.postgres", None)
        with pytest.raises(RuntimeError, match="requires the 'postgres' extra"):
            resolve(postgres_platform(name="pg-missing-extra", conninfo=""))

    def test_resolves_postgres_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaldata.platforms.postgres import PostgresAdapter

        cursor = types.SimpleNamespace(execute=lambda *args, **kwargs: None)
        connection = types.SimpleNamespace(close=lambda: None, cursor=lambda: nullcontext(cursor))
        monkeypatch.setattr("psycopg.connect", lambda *args, **kwargs: connection)
        adapter = resolve(postgres_platform(name="pg-build", conninfo="host=db"))
        try:
            assert isinstance(adapter, PostgresAdapter)
        finally:
            close_all()

    def test_databricks_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "evaldata.platforms.databricks", None)
        with pytest.raises(RuntimeError, match="requires the 'databricks' extra"):
            resolve(databricks_platform(name="dbx-missing-extra", server_hostname="h", http_path="/p"))

    def test_resolves_databricks_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaldata.platforms.databricks import DatabricksAdapter

        cursor = types.SimpleNamespace(execute=lambda *args, **kwargs: None, close=lambda: None)
        connection = types.SimpleNamespace(open=True, close=lambda: None, cursor=lambda: cursor)
        monkeypatch.setattr("databricks.sql.connect", lambda **kwargs: connection)
        monkeypatch.setattr("evaldata.platforms.databricks.Config", lambda host: object())
        adapter = resolve(
            databricks_platform(name="dbx-build", server_hostname="h", http_path="/p", catalog="main", schema="sales")
        )
        try:
            assert isinstance(adapter, DatabricksAdapter)
        finally:
            close_all()

    def test_snowflake_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "evaldata.platforms.snowflake", None)
        with pytest.raises(RuntimeError, match="requires the 'snowflake' extra"):
            resolve(snowflake_platform(name="sf-missing-extra", account="acme-test"))

    def test_resolves_snowflake_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaldata.platforms.snowflake import SnowflakeAdapter

        connection = types.SimpleNamespace(close=lambda: None, is_valid=lambda: True)
        monkeypatch.setattr("snowflake.connector.connect", lambda **kwargs: connection)
        adapter = resolve(snowflake_platform(name="sf-build", account="acme-test", warehouse="COMPUTE_WH"))
        try:
            assert isinstance(adapter, SnowflakeAdapter)
        finally:
            close_all()

    def test_bigquery_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(sys.modules, "evaldata.platforms.bigquery", None)
        with pytest.raises(RuntimeError, match="requires the 'bigquery' extra"):
            resolve(bigquery_platform(name="bq-missing-extra", project="my-proj"))

    def test_resolves_bigquery_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from evaldata.platforms.bigquery import BigQueryAdapter

        monkeypatch.setattr("google.cloud.bigquery.Client", lambda **kwargs: types.SimpleNamespace(close=lambda: None))
        adapter = resolve(bigquery_platform(name="bq-build", project="my-proj", dataset="analytics", location="EU"))
        try:
            assert isinstance(adapter, BigQueryAdapter)
        finally:
            close_all()


@pytest.mark.unit
class TestAcquired:
    def test_yields_a_working_pool_member(self) -> None:
        platform = duckdb_platform(name="acq-work")
        with acquired(platform) as member:
            member.execute("CREATE TABLE t (n INTEGER); INSERT INTO t VALUES (1), (2)")
            result = member.execute("SELECT count(*) AS c FROM t")
        assert isinstance(result, ExecutionSuccess)
        assert result.rows == [{"c": 2}]

    def test_resolve_seed_is_visible_to_an_acquired_member(self) -> None:
        platform = duckdb_platform(name="acq-seed")
        resolve(platform).execute("CREATE TABLE t (n INTEGER); INSERT INTO t VALUES (1), (2), (3)")
        with acquired(platform) as member:
            result = member.execute("SELECT count(*) AS c FROM t")
        assert result.rows == [{"c": 3}]

    def test_resolve_returns_utility_never_a_checkout_member(self) -> None:
        platform = duckdb_platform(name="acq-utility")
        utility = resolve(platform)
        with acquired(platform) as member:
            assert member is not utility

    def test_serial_acquires_reuse_one_member(self) -> None:
        platform = duckdb_platform(name="acq-reuse")
        with acquired(platform) as first:
            first_id = id(first)
        with acquired(platform) as second:
            assert id(second) == first_id

    def test_sqlite_utility_is_distinct_from_a_member_and_shares_data(self) -> None:
        platform = sqlite_platform(name="acq-sqlite-share")
        utility = resolve(platform)
        utility.execute("CREATE TABLE t (n INTEGER)")
        utility.execute("INSERT INTO t VALUES (1), (2)")
        with acquired(platform) as member:
            assert member is not utility
            result = member.execute("SELECT count(*) AS c FROM t")
        assert isinstance(result, ExecutionSuccess)
        assert result.rows == [{"c": 2}]

    def test_member_is_released_even_when_the_block_raises(self) -> None:
        platform = duckdb_platform(name="acq-raise")
        with pytest.raises(RuntimeError, match="boom"), acquired(platform) as first:
            first_id = id(first)
            msg = "boom"
            raise RuntimeError(msg)
        with acquired(platform) as second:
            assert id(second) == first_id

    def test_acquire_from_a_dedicated_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "snowflake.connector.connect",
            lambda **kwargs: types.SimpleNamespace(close=lambda: None, is_valid=lambda: True),
        )
        platform = snowflake_platform(name="acq-dedicated", account="acme-test")
        with acquired(platform) as first, acquired(platform) as second:
            assert first is not second


@pytest.mark.unit
class TestPoolFor:
    def test_returns_the_same_pool_for_a_name(self) -> None:
        platform = duckdb_platform(name="pf-cached")
        assert pool_for(platform) is pool_for(platform)

    def test_same_name_different_config_raises(self) -> None:
        pool_for(duckdb_platform(name="pf-guard", path=":memory:"))
        with pytest.raises(ValueError, match="already bound to a different configuration"):
            pool_for(duckdb_platform(name="pf-guard", path="/tmp/other.duckdb"))

    def test_default_and_explicit_effective_policy_share_a_pool(self) -> None:
        default = duckdb_platform(name="pf-policy")
        explicit = duckdb_platform(name="pf-policy", pool=PoolPolicy(max_size=8))
        assert pool_for(default) is pool_for(explicit)


@pytest.mark.e2e
class TestResolvePostgres:
    def test_resolves_and_executes_postgres(self) -> None:
        from .conftest import _postgres_dsn, connect_postgres

        connect_postgres().close()
        adapter = resolve(postgres_platform(name="registry-pg-e2e", conninfo=_postgres_dsn()))
        try:
            result = adapter.execute("SELECT 1 AS n")
            assert isinstance(result, ExecutionSuccess)
            assert result.rows == [{"n": 1}]
        finally:
            close_all()
