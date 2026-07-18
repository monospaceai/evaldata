"""Unit tests for platform-ref builders and `PlatformRef` -> adapter resolution."""

import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

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
from evaldata.types import PlatformRef


@pytest.mark.unit
class TestRefBuilders:
    def test_duckdb_platform_builds_ref(self) -> None:
        ref = duckdb_platform(name="local", path="/tmp/x.duckdb")
        assert ref == PlatformRef(name="local", kind="duckdb", config={"path": "/tmp/x.duckdb"})

    def test_duckdb_platform_defaults_to_in_memory(self) -> None:
        assert duckdb_platform(name="local").config == {"path": ":memory:"}

    def test_postgres_platform_builds_ref(self) -> None:
        ref = postgres_platform(name="warehouse", conninfo="host=db")
        assert ref == PlatformRef(name="warehouse", kind="postgres", config={"conninfo": "host=db"})

    def test_databricks_platform_builds_ref(self) -> None:
        ref = databricks_platform(name="wh", server_hostname="h.databricks.com", http_path="/sql/1.0/warehouses/abc")
        assert ref == PlatformRef(
            name="wh",
            kind="databricks",
            dialect="databricks",
            config={"server_hostname": "h.databricks.com", "http_path": "/sql/1.0/warehouses/abc"},
        )

    def test_databricks_platform_includes_catalog_and_schema_when_set(self) -> None:
        ref = databricks_platform(name="wh", server_hostname="h", http_path="/p", catalog="main", schema="sales")
        assert ref.config == {"server_hostname": "h", "http_path": "/p", "catalog": "main", "schema": "sales"}

    def test_snowflake_platform_builds_ref(self) -> None:
        ref = snowflake_platform(name="sf", account="acme-test", warehouse="COMPUTE_WH", role="EVALDATA")
        assert ref == PlatformRef(
            name="sf",
            kind="snowflake",
            dialect="snowflake",
            config={"account": "acme-test", "warehouse": "COMPUTE_WH", "role": "EVALDATA"},
        )

    def test_bigquery_platform_builds_ref(self) -> None:
        ref = bigquery_platform(name="bq", project="my-proj")
        assert ref == PlatformRef(name="bq", kind="bigquery", dialect="bigquery", config={"project": "my-proj"})

    def test_bigquery_platform_includes_dataset_and_location_when_set(self) -> None:
        ref = bigquery_platform(name="bq", project="my-proj", dataset="analytics", location="EU")
        assert ref.config == {"project": "my-proj", "dataset": "analytics", "location": "EU"}


@pytest.mark.unit
class TestResolve:
    def test_resolves_and_executes_duckdb(self, tmp_path: Path) -> None:
        db = tmp_path / "t.duckdb"
        adapter = resolve(duckdb_platform(name="local", path=str(db)))
        adapter.execute("CREATE TABLE t (n INTEGER)")
        adapter.execute("INSERT INTO t VALUES (1), (2)")
        result = adapter.execute("SELECT count(*) AS c FROM t")
        assert result.error is None
        assert result.rows == [{"c": 2}]

    def test_same_name_returns_cached_adapter(self) -> None:
        ref = duckdb_platform(name="local")
        assert resolve(ref) is resolve(ref)

    def test_same_name_different_config_raises(self) -> None:
        resolve(duckdb_platform(name="local", path=":memory:"))
        with pytest.raises(ValueError, match="already bound to a different configuration"):
            resolve(duckdb_platform(name="local", path="/tmp/other.duckdb"))

    def test_unsupported_kind_blocked_before_resolution(self) -> None:
        # resolve() dispatches exhaustively over PlatformKind, so it has no "unsupported
        # kind" branch: an unsupported kind can never be constructed into a PlatformRef to
        # hand to resolve in the first place. The boundary is PlatformRef validation.
        with pytest.raises(ValidationError):
            PlatformRef(name="wh", kind="mysql")  # ty: ignore[invalid-argument-type]

    def test_close_all_is_idempotent_when_empty(self) -> None:
        close_all()
        close_all()  # no raise

    def test_postgres_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the 'postgres' extra not being installed: importing the adapter fails.
        monkeypatch.setitem(sys.modules, "evaldata.platforms.postgres", None)
        with pytest.raises(RuntimeError, match="requires the 'postgres' extra"):
            resolve(postgres_platform(name="pg-missing-extra", conninfo=""))

    def test_resolves_postgres_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock psycopg so the registry's postgres dispatch builds an adapter without a live server.
        from evaldata.platforms.postgres import PostgresAdapter

        monkeypatch.setattr("psycopg.connect", lambda *args, **kwargs: types.SimpleNamespace(close=lambda: None))
        adapter = resolve(postgres_platform(name="pg-build", conninfo="host=db"))
        try:
            assert isinstance(adapter, PostgresAdapter)
        finally:
            close_all()

    def test_databricks_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the 'databricks' extra not being installed: importing the adapter fails.
        monkeypatch.setitem(sys.modules, "evaldata.platforms.databricks", None)
        with pytest.raises(RuntimeError, match="requires the 'databricks' extra"):
            resolve(databricks_platform(name="dbx-missing-extra", server_hostname="h", http_path="/p"))

    def test_resolves_databricks_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock the connector so the registry's databricks dispatch builds an adapter without a
        # live workspace.
        from evaldata.platforms.databricks import DatabricksAdapter

        monkeypatch.setattr("databricks.sql.connect", lambda **kwargs: types.SimpleNamespace(close=lambda: None))
        monkeypatch.setattr("evaldata.platforms.databricks.Config", lambda host: object())
        adapter = resolve(
            databricks_platform(name="dbx-build", server_hostname="h", http_path="/p", catalog="main", schema="sales")
        )
        try:
            assert isinstance(adapter, DatabricksAdapter)
        finally:
            close_all()

    def test_snowflake_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the 'snowflake' extra not being installed: importing the adapter fails.
        monkeypatch.setitem(sys.modules, "evaldata.platforms.snowflake", None)
        with pytest.raises(RuntimeError, match="requires the 'snowflake' extra"):
            resolve(snowflake_platform(name="sf-missing-extra", account="acme-test"))

    def test_resolves_snowflake_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock the connector so the registry's snowflake dispatch builds an adapter without a
        # live account.
        from evaldata.platforms.snowflake import SnowflakeAdapter

        monkeypatch.setattr("snowflake.connector.connect", lambda **kwargs: types.SimpleNamespace(close=lambda: None))
        adapter = resolve(snowflake_platform(name="sf-build", account="acme-test", warehouse="COMPUTE_WH"))
        try:
            assert isinstance(adapter, SnowflakeAdapter)
        finally:
            close_all()

    def test_bigquery_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the 'bigquery' extra not being installed: importing the adapter fails.
        monkeypatch.setitem(sys.modules, "evaldata.platforms.bigquery", None)
        with pytest.raises(RuntimeError, match="requires the 'bigquery' extra"):
            resolve(bigquery_platform(name="bq-missing-extra", project="my-proj"))

    def test_resolves_bigquery_through_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Mock the client so the registry's bigquery dispatch builds an adapter without live creds.
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
        assert result.error is None
        assert result.rows == [{"c": 2}]

    def test_resolve_seed_is_visible_to_an_acquired_member(self) -> None:
        # Seeding through `resolve(platform).execute(<seed>)` must be visible to a member
        # acquired for scoring (shared DuckDB parent).
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
            assert id(second) == first_id  # released member reused on the next serial acquire

    def test_sqlite_utility_is_distinct_from_a_member_and_shares_data(self) -> None:
        # SQLite `:memory:` uses a per-name shared-cache database, so the utility and a checkout
        # member are distinct connections that still see each other's data.
        platform = sqlite_platform(name="acq-sqlite-share")
        utility = resolve(platform)
        utility.execute("CREATE TABLE t (n INTEGER)")  # sqlite executes one statement at a time
        utility.execute("INSERT INTO t VALUES (1), (2)")
        with acquired(platform) as member:
            assert member is not utility  # utility is never a checkout member
            result = member.execute("SELECT count(*) AS c FROM t")
        assert result.error is None
        assert result.rows == [{"c": 2}]

    def test_member_is_released_even_when_the_block_raises(self) -> None:
        platform = duckdb_platform(name="acq-raise")
        with pytest.raises(RuntimeError, match="boom"), acquired(platform) as first:
            first_id = id(first)
            msg = "boom"
            raise RuntimeError(msg)
        with acquired(platform) as second:
            assert id(second) == first_id  # the raising block still returned its member

    def test_acquire_from_a_dedicated_pool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A dedicated-connection engine (Snowflake) builds an independent member per checkout;
        # the fake connector keeps this off any live account.
        monkeypatch.setattr("snowflake.connector.connect", lambda **kwargs: types.SimpleNamespace(close=lambda: None))
        platform = snowflake_platform(name="acq-dedicated", account="acme-test")
        with acquired(platform) as first, acquired(platform) as second:
            assert first is not second  # each concurrent checkout is its own connection


@pytest.mark.unit
class TestPoolFor:
    def test_returns_the_same_pool_for_a_name(self) -> None:
        platform = duckdb_platform(name="pf-cached")
        assert pool_for(platform) is pool_for(platform)

    def test_same_name_different_config_raises(self) -> None:
        pool_for(duckdb_platform(name="pf-guard", path=":memory:"))
        with pytest.raises(ValueError, match="already bound to a different configuration"):
            pool_for(duckdb_platform(name="pf-guard", path="/tmp/other.duckdb"))


@pytest.mark.e2e
class TestResolvePostgres:
    """`resolve` builds a live PostgresAdapter through the registry's postgres dispatch."""

    def test_resolves_and_executes_postgres(self) -> None:
        from .conftest import _postgres_dsn, connect_postgres

        connect_postgres().close()  # fail early if Postgres is unreachable
        adapter = resolve(postgres_platform(name="registry-pg-e2e", conninfo=_postgres_dsn()))
        try:
            result = adapter.execute("SELECT 1 AS n")
            assert result.error is None
            assert result.rows == [{"n": 1}]
        finally:
            close_all()
