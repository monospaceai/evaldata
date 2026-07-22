"""End-to-end checks on the Semantic Layer benchmark corpus with a real `mf` run.

Runs every gold query through `mf` and scores the corpus through the full cascade.
"""

import shutil
from pathlib import Path

import pytest

from evaldata.dbt import (
    DbtError,
    MetricCase,
    MetricFirstDecisive,
    MetricQuery,
    MetricResultEquivalence,
    MetricSolverOutput,
    MetricSolverSuccess,
    MetricSpecEquivalence,
    load_dbt_metrics,
    metric_layer_equivalence,
    run,
    run_metric_benchmark,
)
from evaldata.llm import StubLlm
from evaldata.scorers.llm_judge import JudgeReply
from evaldata.types import DuckDBPlatformRef

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "jaffle_sl_bench"
PLATFORM = DuckDBPlatformRef(name="jaffle")
ARTIFACTS = ("manifest.json", "catalog.json", "semantic_manifest.json")


class _GoldSolver:
    """A `MetricSolver` that answers each case with its own gold query."""

    def solve(self, case: MetricCase) -> MetricSolverOutput:
        return MetricSolverSuccess(query=case.gold)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> list[MetricCase]:
    dest = tmp_path_factory.mktemp("jaffle_sl_bench") / "project"
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("target", "logs", "dbt_packages"))
    target = dest / "target"
    target.mkdir()
    for name in ARTIFACTS:
        shutil.copy(dest / "artifacts" / name, target / name)
    cases = load_dbt_metrics(target, platform=PLATFORM, cases=dest / "metric_bench.yml")
    assert not isinstance(cases, DbtError)
    return cases


@pytest.mark.timeout(600)
def test_every_gold_runs(corpus: list[MetricCase]) -> None:
    for case in corpus:
        rows = run(case.gold, case.target_dir, profiles_dir=case.profiles_dir)
        assert not isinstance(rows, DbtError), f"{case.id} did not run: {rows}"
        assert rows, f"{case.id} returned no rows"


@pytest.mark.timeout(600)
def test_corpus_scores_pass_through_the_cascade(corpus: list[MetricCase]) -> None:
    scorers = [metric_layer_equivalence(StubLlm(JudgeReply(reason="unused", score=1.0)))]
    summary = run_metric_benchmark(corpus, _GoldSolver(), scorers=scorers)
    assert summary.total == len(corpus)
    assert summary.passed == summary.total
    assert summary.accuracy == 1.0
    # Each gold resolves to itself, so the spec tier decides; the stub judge is never reached.
    assert all(report.scores[0].basis == "proven" for report in summary.cases)


def test_run_tier_confirms_an_equivalent_reformulation(corpus: list[MetricCase]) -> None:
    case = next(c for c in corpus if c.id == "revenue-by-region")
    candidate = MetricQuery(metrics=["revenue"], group_by=["customer__region"], order_by=["-revenue"])
    cascade = MetricFirstDecisive([MetricSpecEquivalence(), MetricResultEquivalence()])
    score = cascade.score(case, candidate)
    assert score.verdict == "pass"
    assert score.basis == "observed"
