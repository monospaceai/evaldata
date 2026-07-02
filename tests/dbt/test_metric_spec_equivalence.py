"""Tests for MetricFlow canonicalisation and the `MetricSpecEquivalence` scorer.

Canonicalisation resolves a metric query against the committed semantic manifest; it needs the
`dbt-metricflow` toolchain (installed via the `dbt-sl` extra) but touches no warehouse.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evaldata.dbt import DbtError, MetricCase, MetricQuery, MetricSpecEquivalence, canonicalize
from evaldata.dbt.metricflow import CanonicalMetricQuery, _spec_key, group_by_items_by_metric
from evaldata.types import PlatformRef, ScoreResult

pytestmark = pytest.mark.unit

TARGET = Path(__file__).parent / "fixtures" / "jaffle_duckdb" / "artifacts"
PLATFORM = PlatformRef(name="duck", kind="duckdb")


def _case(gold: MetricQuery, *, target: Path = TARGET) -> MetricCase:
    return MetricCase(id="c", input="q", gold=gold, platform=PLATFORM, target_dir=str(target))


def _score(case: MetricCase, query: MetricQuery) -> ScoreResult:
    return MetricSpecEquivalence().score(case, query)


def test_canonicalize_resolves_default_grain() -> None:
    # `metric_time` resolves to the metric's default grain (day), so these are the same query.
    a = canonicalize(MetricQuery(metrics=["revenue"], group_by=["metric_time"]), TARGET)
    b = canonicalize(MetricQuery(metrics=["revenue"], group_by=["metric_time__day"]), TARGET)
    assert isinstance(a, CanonicalMetricQuery)
    assert a == b


def test_canonicalize_captures_where_order_and_limit() -> None:
    where = "{{ Dimension('order_id__is_large_order') }} = true"
    resolved = canonicalize(
        MetricQuery(
            metrics=["revenue"],
            group_by=["metric_time__month"],
            where=[where],
            order_by=["-metric_time__month"],
            limit=5,
        ),
        TARGET,
    )
    assert isinstance(resolved, CanonicalMetricQuery)
    assert resolved.metrics == frozenset({"revenue"})
    assert resolved.limit == 5
    assert resolved.where == frozenset({where})
    assert len(resolved.order_by) == 1
    assert resolved.order_by[0][1] is True  # descending


def test_canonicalize_missing_manifest(tmp_path: Path) -> None:
    result = canonicalize(MetricQuery(metrics=["revenue"]), tmp_path)
    assert isinstance(result, DbtError)
    assert result.kind == "target_not_found"


def test_canonicalize_invalid_query() -> None:
    result = canonicalize(MetricQuery(metrics=["does_not_exist"]), TARGET)
    assert isinstance(result, DbtError)
    assert result.kind == "metric_query_invalid"


def test_canonicalize_without_metricflow(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (
        "metricflow_semantics.model.dbt_manifest_parser",
        "metricflow_semantics.model.semantic_manifest_lookup",
        "metricflow_semantics.query.query_parser",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    result = canonicalize(MetricQuery(metrics=["revenue"]), TARGET)
    assert isinstance(result, DbtError)
    assert result.kind == "metricflow_unavailable"


def test_group_by_items_lists_qualified_dimension_names() -> None:
    items = group_by_items_by_metric(TARGET, ["revenue"])
    assert not isinstance(items, DbtError)
    assert "metric_time" in items["revenue"]
    assert "order_id__is_large_order" in items["revenue"]


def test_group_by_items_prunes_redundant_join_paths() -> None:
    acme = Path(__file__).parent / "fixtures" / "acme_insurance" / "artifacts"
    items = group_by_items_by_metric(acme, ["total_premium"])
    assert not isinstance(items, DbtError)
    # The direct path is kept; the longer re-join to the same dimension (a fanout) is dropped.
    assert "policy__party_identifier_dim" in items["total_premium"]
    assert "policy__party_identifier__party_identifier_dim" not in items["total_premium"]


def test_group_by_items_missing_manifest(tmp_path: Path) -> None:
    result = group_by_items_by_metric(tmp_path, ["revenue"])
    assert isinstance(result, DbtError)
    assert result.kind == "target_not_found"


def test_group_by_items_rejects_unknown_metric() -> None:
    result = group_by_items_by_metric(TARGET, ["does_not_exist"])
    assert isinstance(result, DbtError)
    assert result.kind == "metric_query_invalid"


def test_group_by_items_without_metricflow(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (
        "metricflow_semantics.model.dbt_manifest_parser",
        "metricflow_semantics.model.semantic_manifest_lookup",
        "metricflow_semantic_interfaces.references",
    ):
        monkeypatch.setitem(sys.modules, module, None)
    result = group_by_items_by_metric(TARGET, ["revenue"])
    assert isinstance(result, DbtError)
    assert result.kind == "metricflow_unavailable"


def test_spec_key_includes_grain_and_date_part() -> None:
    spec = SimpleNamespace(
        element_name="ds",
        entity_links=(SimpleNamespace(element_name="order_id"),),
        time_granularity=SimpleNamespace(name="month"),
        date_part=SimpleNamespace(value="year"),
    )
    assert _spec_key(spec) == ("SimpleNamespace", "ds", ("order_id",), "month", "year")


def test_scorer_confirms_equivalent_queries() -> None:
    case = _case(MetricQuery(metrics=["revenue"], group_by=["metric_time"]))
    score = _score(case, MetricQuery(metrics=["revenue"], group_by=["metric_time__day"]))
    assert score.verdict == "pass"
    assert score.basis == "proven"


def test_scorer_is_inconclusive_for_different_queries() -> None:
    case = _case(MetricQuery(metrics=["revenue"], group_by=["metric_time"]))
    score = _score(case, MetricQuery(metrics=["order_count"], group_by=["metric_time"]))
    assert score.verdict == "inconclusive"


def test_scorer_fails_invalid_candidate() -> None:
    # An unresolvable candidate is a decisive fail, not routed onward to the judge.
    case = _case(MetricQuery(metrics=["revenue"]))
    score = _score(case, MetricQuery(metrics=["does_not_exist"]))
    assert score.verdict == "fail"
    assert score.basis == "proven"
    assert score.explanation is not None
    assert score.explanation.startswith("model query:")


def test_scorer_is_inconclusive_when_manifest_missing(tmp_path: Path) -> None:
    case = MetricCase(
        id="c", input="q", gold=MetricQuery(metrics=["revenue"]), platform=PLATFORM, target_dir=str(tmp_path)
    )
    score = _score(case, MetricQuery(metrics=["revenue"]))
    assert score.verdict == "inconclusive"


def test_scorer_is_inconclusive_when_gold_query_is_invalid() -> None:
    case = _case(MetricQuery(metrics=["does_not_exist"]))
    score = _score(case, MetricQuery(metrics=["revenue"]))
    assert score.verdict == "inconclusive"
    assert score.explanation is not None
    assert score.explanation.startswith("gold query:")
