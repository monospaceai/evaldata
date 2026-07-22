"""Hermetic checks on the committed ACME Semantic Layer benchmark corpus.

Resolves each gold query against the committed semantic manifest with `MetricSpecEquivalence`.
Requires `dbt-metricflow`; runs no warehouse query.
"""

from pathlib import Path

import pytest

from evaldata.dbt import DbtError, MetricCase, MetricQuery, MetricSpecEquivalence
from evaldata.dbt._yaml import read_yaml
from evaldata.types import DuckDBPlatformRef

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "acme_insurance"
PLATFORM = DuckDBPlatformRef(name="acme")


def _corpus() -> list[MetricCase]:
    raw = read_yaml(FIXTURE / "acme_bench.yml", not_found="cases_not_found", invalid="cases_invalid")
    if isinstance(raw, DbtError) or not isinstance(raw, list):  # pragma: no cover - the corpus is committed and valid
        msg = "acme_bench.yml is missing or not a list"
        raise RuntimeError(msg)
    return [
        MetricCase(
            id=entry["id"],
            input=entry["question"],
            gold=MetricQuery(
                metrics=entry["metrics"],
                group_by=entry.get("group_by", []),
                where=entry.get("where", []),
                order_by=entry.get("order_by", []),
                limit=entry.get("limit"),
            ),
            platform=PLATFORM,
            target_dir=str(FIXTURE / "artifacts"),
        )
        for entry in raw
    ]


CASES = _corpus()


def test_corpus_is_the_eleven_question_suite() -> None:
    assert len(CASES) == 11
    assert len({case.id for case in CASES}) == 11


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_gold_resolves_against_the_manifest(case: MetricCase) -> None:
    score = MetricSpecEquivalence().score(case, case.gold)
    assert score.verdict == "pass"
    assert score.basis == "proven"
