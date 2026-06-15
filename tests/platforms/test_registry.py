"""Unit tests for platform-ref builders and `PlatformRef` -> adapter resolution."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from dataeval.platforms import duckdb_platform, postgres_platform, resolve
from dataeval.platforms.registry import close_all
from dataeval.types import PlatformRef


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
            PlatformRef(name="wh", kind="snowflake")  # ty: ignore[invalid-argument-type]

    def test_close_all_is_idempotent_when_empty(self) -> None:
        close_all()
        close_all()  # no raise

    def test_postgres_extra_missing_raises_runtime_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Simulate the 'postgres' extra not being installed: importing the adapter fails.
        monkeypatch.setitem(sys.modules, "dataeval.platforms.postgres", None)
        with pytest.raises(RuntimeError, match="requires the 'postgres' extra"):
            resolve(postgres_platform(name="pg-missing-extra", conninfo=""))


@pytest.mark.e2e
class TestResolvePostgres:
    """`resolve` builds a live PostgresAdapter through the registry's postgres dispatch."""

    def test_resolves_and_executes_postgres(self) -> None:
        from .conftest import _postgres_dsn, connect_postgres_or_skip

        connect_postgres_or_skip().close()  # skip unless a Postgres is reachable
        adapter = resolve(postgres_platform(name="registry-pg-e2e", conninfo=_postgres_dsn()))
        try:
            result = adapter.execute("SELECT 1 AS n")
            assert result.error is None
            assert result.rows == [{"n": 1}]
        finally:
            close_all()
