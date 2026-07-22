"""Tests for resolving a dbt profile to a PlatformRef."""

from pathlib import Path

import pytest

from evaldata.dbt.errors import DbtError
from evaldata.dbt.profile import _pg_conninfo, _platform_from_output, platform_from_profile
from evaldata.types import DuckDBPlatformRef, PostgreSQLPlatformRef

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "jaffle_duckdb"


def _project(tmp_path: Path, *, project: str | None = None, profiles: str | None = None) -> Path:
    root = tmp_path / "proj"
    root.mkdir()
    if project is not None:
        (root / "dbt_project.yml").write_text(project, encoding="utf-8")
    if profiles is not None:
        (root / "profiles.yml").write_text(profiles, encoding="utf-8")
    return root


def test_resolves_duckdb_from_fixture() -> None:
    ref = platform_from_profile(FIXTURE)
    assert isinstance(ref, DuckDBPlatformRef)
    assert ref.kind == "duckdb"
    assert Path(ref.config.path).is_absolute()
    assert ref.config.path.endswith("jaffle_duckdb/jaffle.duckdb")


def test_missing_dbt_project(tmp_path: Path) -> None:
    result = platform_from_profile(tmp_path)
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_dbt_project_without_profile(tmp_path: Path) -> None:
    root = _project(tmp_path, project="name: x\n", profiles="x:\n  outputs: {}\n")
    result = platform_from_profile(root)
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_dbt_project_not_a_mapping(tmp_path: Path) -> None:
    root = _project(tmp_path, project="- a\n- b\n")
    result = platform_from_profile(root)
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_malformed_yaml(tmp_path: Path) -> None:
    root = _project(tmp_path, project="a: [unclosed\n")
    result = platform_from_profile(root)
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_missing_profiles(tmp_path: Path) -> None:
    root = _project(tmp_path, project="profile: shop\n")
    result = platform_from_profile(root, profiles_dir=tmp_path / "empty")
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_profile_name_not_in_profiles(tmp_path: Path) -> None:
    root = _project(tmp_path, project="profile: shop\n", profiles="other:\n  outputs: {}\n")
    result = platform_from_profile(root)
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_profile_without_outputs(tmp_path: Path) -> None:
    root = _project(tmp_path, project="profile: shop\n", profiles="shop:\n  target: dev\n")
    result = platform_from_profile(root)
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_target_not_in_outputs(tmp_path: Path) -> None:
    profiles = "shop:\n  target: dev\n  outputs:\n    dev:\n      type: duckdb\n      path: a.db\n"
    root = _project(tmp_path, project="profile: shop\n", profiles=profiles)
    result = platform_from_profile(root, target="prod")
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_no_default_target(tmp_path: Path) -> None:
    profiles = "shop:\n  outputs:\n    dev:\n      type: duckdb\n      path: a.db\n"
    root = _project(tmp_path, project="profile: shop\n", profiles=profiles)
    result = platform_from_profile(root)
    assert isinstance(result, DbtError)
    assert result.kind == "profile_not_found"


def test_platform_from_output_duckdb_memory() -> None:
    ref = _platform_from_output("n", {"type": "duckdb"}, Path("/proj"))
    assert isinstance(ref, DuckDBPlatformRef)
    assert ref.config.path == ":memory:"


def test_platform_from_output_duckdb_absolute_path() -> None:
    ref = _platform_from_output("n", {"type": "duckdb", "path": "/abs/x.db"}, Path("/proj"))
    assert isinstance(ref, DuckDBPlatformRef)
    assert ref.config.path == "/abs/x.db"


def test_platform_from_output_postgres() -> None:
    ref = _platform_from_output("n", {"type": "postgres", "host": "h", "dbname": "d", "user": "u"}, Path("/proj"))
    assert isinstance(ref, PostgreSQLPlatformRef)
    assert ref.kind == "postgres"
    assert ref.config.conninfo == "host=h dbname=d user=u"


def test_platform_from_output_unsupported_adapter() -> None:
    result = _platform_from_output("n", {"type": "snowflake"}, Path("/proj"))
    assert isinstance(result, DbtError)
    assert result.kind == "unsupported_adapter"


def test_pg_conninfo_includes_present_fields_only() -> None:
    assert _pg_conninfo({"host": "h", "port": 5432, "dbname": "d", "user": "u"}) == "host=h port=5432 dbname=d user=u"
