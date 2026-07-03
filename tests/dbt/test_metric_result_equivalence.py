"""Unit tests for `MetricResultEquivalence`, with the `mf` run stubbed at the module boundary."""

import pytest

from evaldata.dbt import DbtError, MetricCase, MetricQuery, MetricResultEquivalence
from evaldata.types import PlatformRef

pytestmark = pytest.mark.unit

PLATFORM = PlatformRef(name="duck", kind="duckdb")


def _case() -> MetricCase:
    return MetricCase(id="c", input="q", gold=MetricQuery(metrics=["revenue"]), platform=PLATFORM, target_dir="t")


def _stub_run(monkeypatch: pytest.MonkeyPatch, results: list[object]) -> None:
    calls = iter(results)
    monkeypatch.setattr("evaldata.dbt.metric_result_equivalence.run", lambda *a, **k: next(calls))


def _score(monkeypatch, candidate, gold, **kwargs):
    _stub_run(monkeypatch, [candidate, gold])
    return MetricResultEquivalence(**kwargs).score(_case(), MetricQuery(metrics=["revenue"]))


def test_pass_when_rows_match(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"metric_time__month": "2024-01", "revenue": "50"}]
    score = _score(monkeypatch, rows, list(reversed(rows)))
    assert score.verdict == "pass"
    assert score.basis == "observed"


def test_fail_when_rows_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _score(monkeypatch, [{"revenue": "50"}], [{"revenue": "99"}])
    assert score.verdict == "fail"
    assert score.basis == "observed"


def test_pass_when_column_name_differs_but_values_match(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _score(monkeypatch, [{"total_policies": "2"}], [{"number_of_policies": "2"}])
    assert score.verdict == "pass"


def test_pass_within_numeric_formatting(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _score(monkeypatch, [{"revenue": "98000"}], [{"revenue": "98000.0"}])
    assert score.verdict == "pass"


def test_pass_with_redundant_extra_column(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = [{"region": "N", "name": "Alice", "revenue": "100"}]
    gold = [{"region": "N", "revenue": "100"}]
    assert _score(monkeypatch, candidate, gold).verdict == "pass"


def test_fail_when_extra_column_is_not_redundant(monkeypatch: pytest.MonkeyPatch) -> None:
    candidate = [{"region": "N", "tier": "gold", "revenue": "50"}, {"region": "N", "tier": "silver", "revenue": "50"}]
    gold = [{"region": "N", "revenue": "100"}]
    assert _score(monkeypatch, candidate, gold).verdict == "fail"


def test_fail_when_candidate_is_missing_a_grouping(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _score(monkeypatch, [{"revenue": "100"}], [{"region": "N", "revenue": "100"}])
    assert score.verdict == "fail"


def test_inconclusive_when_model_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run(monkeypatch, [DbtError(kind="metric_query_invalid", message="boom")])
    score = MetricResultEquivalence().score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "inconclusive"
    assert score.explanation is not None
    assert score.explanation.startswith("model query:")


def test_fail_when_model_query_fails_under_strict(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run(monkeypatch, [DbtError(kind="metric_query_invalid", message="boom")])
    score = MetricResultEquivalence(on_error="fail").score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "fail"
    assert score.basis == "observed"


def test_inconclusive_when_gold_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run(monkeypatch, [[{"revenue": "50"}], DbtError(kind="metricflow_unavailable", message="no mf")])
    score = MetricResultEquivalence().score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "inconclusive"
    assert score.explanation is not None
    assert score.explanation.startswith("gold query:")
