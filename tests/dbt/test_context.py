"""Tests for `DbtContext` normalisation and schema rendering."""

import json
import shutil
from pathlib import Path

import pytest

from evaldata.dbt import (
    Column,
    DbtContext,
    DbtError,
    Relation,
    SchemaContext,
    TableSchema,
)

pytestmark = pytest.mark.unit

FIXTURE_ARTIFACTS = Path(__file__).parent / "fixtures" / "jaffle_duckdb" / "artifacts"


@pytest.fixture
def ctx() -> DbtContext:
    built = DbtContext.from_target_dir(FIXTURE_ARTIFACTS)
    assert isinstance(built, DbtContext)
    return built


def test_from_target_dir_passes_through_errors(tmp_path: Path) -> None:
    result = DbtContext.from_target_dir(tmp_path)
    assert isinstance(result, DbtError)
    assert result.kind == "target_not_found"


def test_builds_context_from_fusion_v20_manifest(tmp_path: Path) -> None:
    # dbt Fusion emits schema v20; verify the reader accepts higher version tokens.
    manifest = json.loads((FIXTURE_ARTIFACTS / "manifest.json").read_text(encoding="utf-8"))
    manifest["metadata"]["dbt_schema_version"] = "https://schemas.getdbt.com/dbt/manifest/v20.json"
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    built = DbtContext.from_target_dir(target)
    assert isinstance(built, DbtContext)
    assert built.schema_version() == "v20"
    assert built.model("customers") is not None


def test_models_and_sources_are_normalised(ctx: DbtContext) -> None:
    assert {m.name for m in ctx._models} == {"stg_customers", "stg_orders", "customers"}
    assert {(s.source_name, s.name) for s in ctx.sources()} == {("jaffle", "raw_customers"), ("jaffle", "raw_orders")}
    assert ctx.schema_version() == "v12"


def test_models_returns_all_models_in_order(ctx: DbtContext) -> None:
    assert [m.name for m in ctx.models()] == ["stg_customers", "stg_orders", "customers"]


def test_tests_returns_model_tests(ctx: DbtContext) -> None:
    assert {(t.name, t.model, t.column) for t in ctx.tests()} == {
        ("unique", "customers", "customer_id"),
        ("not_null", "customers", "customer_id"),
    }


def test_tests_skip_singular_and_unattached(tmp_path: Path) -> None:
    manifest = {
        "metadata": {"dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json"},
        "nodes": {
            "model.p.m": {
                "resource_type": "model",
                "name": "m",
                "database": "db",
                "schema": "sc",
                "alias": "m",
                "relation_name": '"db"."sc"."m"',
                "columns": {},
                "compiled_code": "select 1",
                "description": "m",
            },
            "test.p.generic": {
                "resource_type": "test",
                "test_metadata": {"name": "not_null"},
                "column_name": "id",
                "attached_node": "model.p.m",
            },
            "test.p.singular": {"resource_type": "test", "column_name": None, "attached_node": "model.p.m"},
            "test.p.unattached": {
                "resource_type": "test",
                "test_metadata": {"name": "unique"},
                "column_name": "id",
                "attached_node": "model.p.unknown",
            },
        },
        "sources": {},
    }
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    ctx = DbtContext.from_target_dir(target)
    assert isinstance(ctx, DbtContext)
    # The singular test (no test_metadata) and the test on an unknown model are dropped.
    assert [(t.name, t.model, t.column) for t in ctx.tests()] == [("not_null", "m", "id")]


def test_model_addressable_by_name_and_uid(ctx: DbtContext) -> None:
    by_name = ctx.model("customers")
    by_uid = ctx.model("model.jaffle_duckdb.customers")
    assert by_name is not None
    assert by_name is by_uid


def test_missing_model_returns_none(ctx: DbtContext) -> None:
    assert ctx.model("nope") is None
    assert ctx.compiled_sql("nope") is None
    assert ctx.relation("nope") is None


def test_compiled_sql_and_relation(ctx: DbtContext) -> None:
    assert "select" in (ctx.compiled_sql("customers") or "").lower()
    relation = ctx.relation("customers")
    assert relation == Relation(
        database="jaffle", schema="main", identifier="customers", quoted='"jaffle"."main"."customers"'
    )
    assert str(relation) == '"jaffle"."main"."customers"'


def test_columns_use_catalog_types_with_manifest_descriptions(ctx: DbtContext) -> None:
    columns = {c.name: c for c in ctx.model("customers").columns}
    # All four columns come from the catalog; only two are documented in the manifest.
    assert set(columns) == {"customer_id", "customer_name", "order_count", "lifetime_value"}
    assert columns["customer_id"].type == "INTEGER"
    assert columns["customer_id"].description == "Surrogate key for the customer."
    # customer_name is catalog-typed but has no manifest description.
    assert columns["customer_name"].type == "VARCHAR"
    assert columns["customer_name"].description is None


def test_degrades_to_manifest_columns_without_catalog(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    shutil.copy(FIXTURE_ARTIFACTS / "manifest.json", target / "manifest.json")
    built = DbtContext.from_target_dir(target)
    assert isinstance(built, DbtContext)

    columns = {c.name: c for c in built.model("customers").columns}
    assert set(columns) == {"customer_id", "lifetime_value"}
    assert columns["customer_id"].type is None
    assert columns["customer_id"].description == "Surrogate key for the customer."
    # The fixture sources have no manifest-documented columns.
    assert built.sources()[0].columns == ()


def test_tables_returns_sources_then_models(ctx: DbtContext) -> None:
    tables = ctx.tables()
    assert [t.name for t in tables] == ["raw_customers", "raw_orders", "stg_customers", "stg_orders", "customers"]
    assert all(isinstance(t, TableSchema) for t in tables)


def test_schema_context_default_includes_everything(ctx: DbtContext) -> None:
    names = {t.name for t in ctx.schema_context().tables}
    assert names == {"raw_customers", "raw_orders", "stg_customers", "stg_orders", "customers"}


def test_schema_context_can_exclude_sources(ctx: DbtContext) -> None:
    names = {t.name for t in ctx.schema_context(include_sources=False).tables}
    assert names == {"stg_customers", "stg_orders", "customers"}


def test_schema_context_can_exclude_models(ctx: DbtContext) -> None:
    names = {t.name for t in ctx.schema_context(include_models=False).tables}
    assert names == {"raw_customers", "raw_orders"}


def test_schema_context_select_filters_by_name(ctx: DbtContext) -> None:
    names = {t.name for t in ctx.schema_context(select=["customers"]).tables}
    assert names == {"customers"}


def test_schema_context_renders_create_table(ctx: DbtContext) -> None:
    text = ctx.schema_context(select=["customers"]).as_text()
    assert "-- Customer dimension enriched with order activity." in text
    assert 'CREATE TABLE "jaffle"."main"."customers" (' in text
    assert "  customer_id INTEGER,  -- Surrogate key for the customer." in text
    assert text.rstrip().endswith(");")


def test_empty_schema_context_renders_empty_string(ctx: DbtContext) -> None:
    assert ctx.schema_context(select=["does-not-exist"]).as_text() == ""


def test_as_text_handles_descriptions_types_and_empty_tables() -> None:
    context = SchemaContext(
        tables=(
            TableSchema(
                name="t1",
                relation=Relation("db", "sc", "t1", '"db"."sc"."t1"'),
                columns=(Column("a", "INT", "the a"), Column("b", None, None), Column("c", "TEXT", None)),
                description="table one",
            ),
            TableSchema(
                name="t2",
                relation=Relation("db", "sc", "t2", '"db"."sc"."t2"'),
                columns=(),
                description=None,
            ),
        )
    )
    text = context.as_text()
    assert text == (
        "-- table one\n"
        'CREATE TABLE "db"."sc"."t1" (\n'
        "  a INT,  -- the a\n"
        "  b,\n"
        "  c TEXT\n"
        ");\n"
        "\n"
        'CREATE TABLE "db"."sc"."t2" (\n'
        ");"
    )
