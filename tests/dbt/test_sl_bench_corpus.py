"""Hermetic checks on the committed Semantic Layer benchmark corpus.

Resolves every gold query against the committed semantic manifest with `MetricSpecEquivalence`.
Requires `dbt-metricflow`; runs no warehouse query.
"""

from pathlib import Path

import pytest

from evaldata.dbt import (
    DbtError,
    MetricCase,
    MetricQuery,
    MetricSpecEquivalence,
    load_dbt_metrics,
)
from evaldata.types import PlatformRef

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "jaffle_sl_bench"
PLATFORM = PlatformRef(name="jaffle", kind="duckdb")


def _corpus() -> list[MetricCase]:
    cases = load_dbt_metrics(FIXTURE / "artifacts", platform=PLATFORM, cases=FIXTURE / "metric_bench.yml")
    if isinstance(cases, DbtError):  # pragma: no cover - the corpus is committed and valid
        raise RuntimeError(cases.message)
    return cases


CASES = _corpus()


def test_corpus_is_a_stable_size() -> None:
    assert len(CASES) >= 30
    assert len({case.id for case in CASES}) == len(CASES)


def test_cases_carry_semantic_context() -> None:
    assert all(case.sl_context for case in CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_gold_resolves_against_the_manifest(case: MetricCase) -> None:
    score = MetricSpecEquivalence().score(case, case.gold)
    assert score.verdict == "pass"
    assert score.basis == "proven"


def test_spec_tier_does_not_confirm_a_different_query() -> None:
    case = next(c for c in CASES if c.id == "total-revenue")
    score = MetricSpecEquivalence().score(case, MetricQuery(metrics=["order_count"]))
    assert score.verdict == "inconclusive"
