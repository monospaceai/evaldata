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


def test_pass_when_rows_match(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{"metric_time__month": "2024-01", "revenue": "50"}]
    _stub_run(monkeypatch, [rows, list(reversed(rows))])
    score = MetricResultEquivalence().score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "pass"
    assert score.basis == "observed"


def test_fail_when_rows_differ(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run(monkeypatch, [[{"revenue": "50"}], [{"revenue": "99"}]])
    score = MetricResultEquivalence().score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "fail"
    assert score.basis == "observed"


def test_inconclusive_when_model_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run(monkeypatch, [DbtError(kind="metric_query_invalid", message="boom")])
    score = MetricResultEquivalence().score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "inconclusive"
    assert score.explanation is not None
    assert score.explanation.startswith("model query:")


def test_inconclusive_when_gold_query_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_run(monkeypatch, [[{"revenue": "50"}], DbtError(kind="metricflow_unavailable", message="no mf")])
    score = MetricResultEquivalence().score(_case(), MetricQuery(metrics=["revenue"]))
    assert score.verdict == "inconclusive"
    assert score.explanation is not None
    assert score.explanation.startswith("gold query:")
