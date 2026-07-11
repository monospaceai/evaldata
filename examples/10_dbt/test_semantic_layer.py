"""dbt Semantic Layer (MetricFlow) evals against the jaffle-shop project, run locally on DuckDB.

`load_dbt_metrics` builds one `MetricCase` per question from a metric cases file; each gold is a
`MetricQuery` — the metrics to compute and how to group, filter, and order them. Candidates are
scored by three checks, cheapest first, stopping at the first verdict:

- `MetricSpecEquivalence` resolves both queries through MetricFlow and confirms a match from their
  resolved form, without running anything. It proves, for example, that grouping by `metric_time`
  (whose default grain is a day) equals grouping by `metric_time__day`.
- `MetricResultEquivalence` runs both queries through `mf` and compares the result rows, so a
  reformulation that returns the same rows is confirmed even when the resolved forms differ.
- `metric_layer_equivalence(...)` is the cost-ordered cascade of both plus an LLM judge. The judge
  here is a `StubLlm`, so the file runs offline with no model or warehouse service.

Running `mf` needs the `dbt-metricflow` toolchain with a DuckDB adapter (the `dbt-sl` extra plus
`dbt-duckdb`); resolving a query needs only the semantic manifest.
"""

import shutil
from pathlib import Path

import pytest

from evaldata.dbt import (
    DbtError,
    MetricCase,
    MetricFirstDecisive,
    MetricLayerSolver,
    MetricQuery,
    MetricResultEquivalence,
    MetricSpecEquivalence,
    load_dbt_metrics,
    metric_layer_equivalence,
    run_metric_benchmark,
)
from evaldata.llm import StubLlm
from evaldata.scorers.llm_judge import JudgeReply
from evaldata.types import PlatformRef

_PROJECT = Path(__file__).parent / "jaffle"
_CASES = Path(__file__).parent / "metric_cases.yml"
_PLATFORM = PlatformRef(name="jaffle", kind="duckdb")
_ARTIFACTS = ("manifest.json", "catalog.json", "semantic_manifest.json")


@pytest.fixture
def cases(tmp_path: Path) -> dict[str, MetricCase]:
    """Copy the project to a temp dir, seed `target/` with the artifacts, and load the cases by id.

    Args:
        tmp_path: A temp directory the project is copied into so its files are never mutated.

    Returns:
        The loaded Semantic Layer cases, keyed by case id.
    """
    project = tmp_path / "jaffle"
    shutil.copytree(_PROJECT, project)
    target = project / "target"
    target.mkdir()
    for name in _ARTIFACTS:
        shutil.copy(project / "artifacts" / name, target / name)
    loaded = load_dbt_metrics(target, platform=_PLATFORM, cases=_CASES)
    assert not isinstance(loaded, DbtError)
    return {case.id: case for case in loaded}


def test_default_grain_proven_without_running(cases: dict[str, MetricCase]) -> None:
    """Grouping by `metric_time` resolves to its default grain, so `metric_time__day` is proven equal — no query runs."""
    case = cases["revenue-by-day"]
    score = MetricSpecEquivalence().score(case, MetricQuery(metrics=["revenue"], group_by=["metric_time__day"]))
    assert score.verdict == "pass"
    assert score.basis == "proven"


@pytest.mark.timeout(300)
def test_reformulation_confirmed_by_results(cases: dict[str, MetricCase]) -> None:
    """Adding an ordering the resolver treats as different still passes: the run tier confirms the same rows."""
    case = cases["revenue-by-region"]
    candidate = MetricQuery(metrics=["revenue"], group_by=["customer__region"], order_by=["-revenue"])
    cascade = MetricFirstDecisive([MetricSpecEquivalence(), MetricResultEquivalence()])
    score = cascade.score(case, candidate)
    assert score.verdict == "pass"
    assert score.basis == "observed"


def test_large_order_filter_resolves(cases: dict[str, MetricCase]) -> None:
    """A filter on `order_id__is_large_order` is a valid query the spec tier resolves and confirms."""
    case = cases["revenue-large-orders"]
    score = MetricSpecEquivalence().score(case, case.gold)
    assert score.verdict == "pass"
    assert score.basis == "proven"


def _answer(prompt: str, _response_format: object) -> MetricQuery:
    """Answer three of the questions correctly and deliberately misread the fourth.

    Args:
        prompt: The solver prompt, which carries the question text.
        _response_format: The requested structured-output type (unused).

    Returns:
        The metric query the stubbed solver answers the question with.
    """
    if "each month" in prompt:
        return MetricQuery(metrics=["revenue"], group_by=["metric_time__month"])
    if "customer region" in prompt:
        return MetricQuery(metrics=["revenue"], group_by=["customer__region"])
    if "large orders" in prompt:
        return MetricQuery(metrics=["revenue"], where=["{{ Dimension('order_id__is_large_order') }} = true"])
    # Misreads the daily question as monthly — the one miss the benchmark catches.
    return MetricQuery(metrics=["revenue"], group_by=["metric_time__month"])


@pytest.mark.timeout(300)
def test_benchmark_reports_accuracy(cases: dict[str, MetricCase]) -> None:
    """Three questions answered correctly and one deliberately misread aggregate to 0.75."""
    solver = MetricLayerSolver(model=StubLlm(_answer))
    # The judge is the last tier; here spec or run decides every case, so the stub grader is never reached.
    scorers = [metric_layer_equivalence(StubLlm(JudgeReply(reason="unused", score=0.0)))]
    summary = run_metric_benchmark(list(cases.values()), solver, scorers=scorers)
    assert summary.total == 4
    assert summary.passed == 3
    assert summary.accuracy == 0.75
