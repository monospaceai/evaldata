"""End-to-end Semantic Layer eval: a real `mf` run over the committed fixture DuckDB.

Copies the fixture to a temp dir so the committed files are never mutated, and seeds `target/` with
the committed semantic manifest so `mf` can run without rebuilding the project.
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
    assert_metric_eval,
    run,
    run_metric_benchmark,
)
from evaldata.types import DuckDBPlatformRef


class _StubSolver:
    """A `MetricSolver` that always returns the same candidate query."""

    def __init__(self, query: MetricQuery) -> None:
        self._query = query

    def solve(self, case: "MetricCase") -> MetricSolverOutput:
        return MetricSolverSuccess(query=self._query)


pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "jaffle_duckdb"
PLATFORM = DuckDBPlatformRef(name="jaffle")


def _target_dir(tmp_path: Path) -> Path:
    dest = tmp_path / "jaffle"
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("target", "logs", "dbt_packages"))
    target = dest / "target"
    target.mkdir()
    shutil.copy(dest / "artifacts" / "semantic_manifest.json", target / "semantic_manifest.json")
    return target


def _case(target: Path, gold: MetricQuery, *, id: str = "c") -> MetricCase:
    return MetricCase(id=id, input="q", gold=gold, platform=PLATFORM, target_dir=str(target))


def test_run_executes_query(tmp_path: Path) -> None:
    rows = run(MetricQuery(metrics=["revenue"], group_by=["metric_time__month"]), _target_dir(tmp_path))
    assert not isinstance(rows, DbtError)
    assert rows
    assert set(rows[0]) == {"metric_time__month", "revenue"}


def test_run_reports_invalid_query(tmp_path: Path) -> None:
    result = run(MetricQuery(metrics=["does_not_exist"]), _target_dir(tmp_path))
    assert isinstance(result, DbtError)
    assert result.kind == "metric_query_invalid"


def test_result_equivalence_passes_for_same_rows(tmp_path: Path) -> None:
    target = _target_dir(tmp_path)
    case = _case(target, MetricQuery(metrics=["revenue"], group_by=["metric_time__month"]))
    score = MetricResultEquivalence().score(case, MetricQuery(metrics=["revenue"], group_by=["metric_time__month"]))
    assert score.verdict == "pass"
    assert score.basis == "observed"


def test_result_equivalence_fails_for_different_rows(tmp_path: Path) -> None:
    target = _target_dir(tmp_path)
    case = _case(target, MetricQuery(metrics=["revenue"], group_by=["metric_time__month"]))
    score = MetricResultEquivalence().score(case, MetricQuery(metrics=["order_count"], group_by=["metric_time__month"]))
    assert score.verdict == "fail"


def test_cascade_confirms_by_spec_before_running(tmp_path: Path) -> None:
    target = _target_dir(tmp_path)
    case = _case(target, MetricQuery(metrics=["revenue"], group_by=["metric_time"]))
    cascade = MetricFirstDecisive([MetricSpecEquivalence(), MetricResultEquivalence()])
    # `metric_time__day` is the resolved default grain of `metric_time`, so the spec tier confirms.
    score = cascade.score(case, MetricQuery(metrics=["revenue"], group_by=["metric_time__day"]))
    assert score.verdict == "pass"
    assert score.basis == "proven"
    assert score.metadata["first_decisive"] == [
        {"scorer": "metric_spec_equivalence", "passed": True, "verdict": "pass"}
    ]


def test_run_metric_benchmark_e2e(tmp_path: Path) -> None:
    target = _target_dir(tmp_path)
    right = _StubSolver(MetricQuery(metrics=["revenue"], group_by=["metric_time__month"]))
    cases = [
        _case(target, MetricQuery(metrics=["revenue"], group_by=["metric_time__month"]), id="right"),
        _case(target, MetricQuery(metrics=["order_count"], group_by=["metric_time__month"]), id="wrong"),
    ]
    summary = run_metric_benchmark(cases, right, scorers=[MetricResultEquivalence()])
    assert summary.total == 2
    assert summary.passed == 1
    assert summary.accuracy == 0.5


def test_assert_metric_eval_e2e(tmp_path: Path) -> None:
    target = _target_dir(tmp_path)
    case = _case(target, MetricQuery(metrics=["revenue"], group_by=["metric_time"]))
    solver = _StubSolver(MetricQuery(metrics=["revenue"], group_by=["metric_time__day"]))
    assert_metric_eval(case, solver, scorers=[MetricSpecEquivalence()])
