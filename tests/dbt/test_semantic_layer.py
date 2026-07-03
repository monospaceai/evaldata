"""Tests for `DbtContext` semantic-layer parsing and prompt rendering."""

import json
from pathlib import Path

import pytest

from evaldata.dbt import (
    DbtContext,
    DbtError,
    Dimension,
    Entity,
    Metric,
    Relation,
    SemanticLayerContext,
    SemanticModel,
)

pytestmark = pytest.mark.unit

FIXTURE_ARTIFACTS = Path(__file__).parent / "fixtures" / "jaffle_duckdb" / "artifacts"
_RELATION = Relation(database="db", schema="sc", identifier="m", quoted='"db"."sc"."m"')

MANIFEST_HEADER = {
    "metadata": {"dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json"},
    "nodes": {},
    "sources": {},
}


@pytest.fixture
def ctx() -> DbtContext:
    built = DbtContext.from_target_dir(FIXTURE_ARTIFACTS)
    assert isinstance(built, DbtContext)
    return built


def _write_target(tmp_path: Path, *, semantic: object | None = None) -> Path:
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text(json.dumps(MANIFEST_HEADER), encoding="utf-8")
    if semantic is not None:
        (target / "semantic_manifest.json").write_text(json.dumps(semantic), encoding="utf-8")
    return target


def test_parses_semantic_model_from_fixture(ctx: DbtContext) -> None:
    models = ctx.semantic_models()
    assert [m.name for m in models] == ["orders"]
    orders = models[0]
    assert orders.description == "Order fact, one row per placed order."
    assert orders.relation == Relation(
        database="jaffle", schema="main", identifier="stg_orders", quoted='"jaffle"."main"."stg_orders"'
    )
    assert orders.entities == (
        Entity(name="order_id", type="primary", expr=None),
        Entity(name="customer", type="foreign", expr="customer_id"),
    )
    assert orders.dimensions == (
        Dimension(
            name="ordered_at",
            type="time",
            expr="cast(order_date as date)",
            granularity="day",
            description="The date the order was placed.",
        ),
        Dimension(
            name="is_large_order",
            type="categorical",
            expr="amount >= 50",
            granularity=None,
            description="Whether the order total is at least 50.",
        ),
    )
    assert {(m.name, m.agg) for m in orders.measures} == {("order_total", "sum"), ("order_count", "sum")}


def test_parses_metrics_from_fixture(ctx: DbtContext) -> None:
    assert {(m.name, m.type) for m in ctx.metrics()} == {
        ("revenue", "simple"),
        ("order_count", "simple"),
        ("average_order_value", "ratio"),
        ("revenue_doubled", "derived"),
    }
    revenue = next(m for m in ctx.metrics() if m.name == "revenue")
    assert revenue.label == "Revenue"
    assert revenue.description == "Sum of order amounts."


def test_dimensions_dedupes_by_name(ctx: DbtContext) -> None:
    assert [d.name for d in ctx.dimensions()] == ["ordered_at", "is_large_order"]


def test_sl_context_renders_fixture(ctx: DbtContext) -> None:
    text = ctx.sl_context().as_text()
    assert "  revenue (simple) -- Sum of order amounts." in text
    assert "Semantic model: orders" in text
    assert "  entities: order_id (primary), customer (foreign)" in text
    assert "    ordered_at (time, day) -- The date the order was placed." in text
    assert "    is_large_order (categorical) -- Whether the order total is at least 50." in text
    assert "  measures: order_total (sum), order_count (sum)" in text


def test_project_without_semantic_layer(tmp_path: Path) -> None:
    built = DbtContext.from_target_dir(_write_target(tmp_path))
    assert isinstance(built, DbtContext)
    assert built.semantic_models() == []
    assert built.metrics() == []
    assert built.dimensions() == []
    assert built.sl_context().as_text() == ""


def test_invalid_semantic_manifest_is_an_error(tmp_path: Path) -> None:
    built = DbtContext.from_target_dir(_write_target(tmp_path, semantic=["not", "an", "object"]))
    assert isinstance(built, DbtError)
    assert built.kind == "artifact_invalid"


def test_parses_optional_fields_and_dedupes_across_models(tmp_path: Path) -> None:
    # A bare metric (no label/description), an entity with no expr, a measure with no description,
    # a dimension with no description or type_params, and a name shared across two models.
    semantic = {
        "semantic_models": [
            {
                "name": "a",
                "node_relation": {
                    "database": "db",
                    "schema_name": "sc",
                    "alias": "a",
                    "relation_name": '"db"."sc"."a"',
                },
                "entities": [{"name": "shared", "type": "primary"}],
                "dimensions": [{"name": "region", "type": "categorical"}],
                "measures": [{"name": "rows", "agg": "count"}],
            },
            {
                "name": "b",
                "node_relation": {
                    "database": "db",
                    "schema_name": "sc",
                    "alias": "b",
                    "relation_name": '"db"."sc"."b"',
                },
                "entities": [{"name": "shared", "type": "foreign"}],
                "dimensions": [{"name": "region", "type": "categorical"}],
                "measures": [],
            },
        ],
        "metrics": [{"name": "m", "type": "simple"}],
    }
    built = DbtContext.from_target_dir(_write_target(tmp_path, semantic=semantic))
    assert isinstance(built, DbtContext)

    assert built.metrics() == [Metric(name="m", type="simple", label=None, description=None)]
    model_a = built.semantic_models()[0]
    assert model_a.entities[0].expr is None
    assert model_a.measures[0].description is None
    assert model_a.dimensions[0] == Dimension(
        name="region", type="categorical", expr=None, granularity=None, description=None
    )
    # The flat `dimensions()` accessor dedupes by name...
    assert [d.name for d in built.dimensions()] == ["region"]
    # ...but the prompt context keeps each model's dimensions distinct, so a same-named dimension
    # on two models stays disambiguable.
    text = built.sl_context().as_text()
    assert "Semantic model: a" in text
    assert "Semantic model: b" in text
    assert text.count("region (categorical)") == 2


def test_sl_context_renders_optional_fields_and_empty() -> None:
    empty_model = SemanticModel(name="m", description=None, relation=_RELATION, entities=(), dimensions=(), measures=())
    context = SemanticLayerContext(
        metrics=(Metric(name="rev", type="simple", label=None, description=None),),
        semantic_models=(empty_model,),
    )
    assert context.as_text() == "Metrics:\n  rev (simple)\n\nSemantic model: m"
    assert SemanticLayerContext(metrics=(), semantic_models=()).as_text() == ""
