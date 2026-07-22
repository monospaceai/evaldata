"""End-to-end reproduction of the ACME Semantic Layer benchmark on local DuckDB.

For each of the 11 questions, runs the gold MetricFlow query through `mf` and asserts its rows
match the gold SQL on the same warehouse, comparing metric values within a 1e-5 tolerance.
"""

import os
import shutil
import subprocess
from pathlib import Path

import duckdb
import pytest

from evaldata.dbt import (
    DbtError,
    MetricCase,
    MetricSolverOutput,
    MetricSolverSuccess,
    load_dbt_metrics,
    metric_layer_equivalence,
    run,
    run_metric_benchmark,
)
from evaldata.dbt._yaml import read_yaml
from evaldata.llm import StubLlm
from evaldata.scorers.llm_judge import JudgeReply
from evaldata.types import DuckDBPlatformRef

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "acme_insurance"
PLATFORM = DuckDBPlatformRef(name="acme")


class _GoldSolver:
    """A `MetricSolver` that answers each case with its own gold query."""

    def solve(self, case: MetricCase) -> MetricSolverOutput:
        return MetricSolverSuccess(query=case.gold)


def _measure(value: object) -> float | None:
    """The numeric metric value, or `None` for a SQL NULL or empty cell; raises on non-numeric."""
    if value is None or value == "":
        return None
    return float(value)  # type: ignore[arg-type]


def _close(got: float | None, want: float | None) -> bool:
    if got is None or want is None:
        return got is want
    return abs(got - want) <= 1e-5 + 1e-5 * abs(want)


def _mf_index(rows: list[dict[str, str]], metric: str) -> dict[tuple[str, ...], float | None]:
    """Map each row's group-by values to its metric value, keyed by the non-metric columns in order."""
    index: dict[tuple[str, ...], float | None] = {}
    for row in rows:
        assert metric in row, f"metric column {metric!r} missing from {list(row)}"
        key = tuple(str(value) for column, value in row.items() if column != metric)
        assert key not in index, f"duplicate group key {key}"
        index[key] = _measure(row[metric])
    return index


def _sql_index(rows: list[tuple[object, ...]]) -> dict[tuple[str, ...], float | None]:
    """Map each SQL row's leading columns to its final column, the measure."""
    index: dict[tuple[str, ...], float | None] = {}
    for row in rows:
        key = tuple(str(value) for value in row[:-1])
        assert key not in index, f"duplicate group key {key}"
        index[key] = _measure(row[-1])
    return index


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    dest = tmp_path_factory.mktemp("acme") / "project"
    shutil.copytree(
        FIXTURE, dest, ignore=shutil.ignore_patterns("target", "logs", "dbt_packages", "artifacts", "acme.duckdb")
    )
    dbt = shutil.which("dbt")
    assert dbt is not None, "dbt is not on PATH; install the 'fixtures' group"
    env = {**os.environ, "DBT_PROFILES_DIR": str(dest)}
    for command in (["seed"], ["build"], ["parse"]):
        result = subprocess.run(
            [dbt, *command, "--profiles-dir", str(dest)], cwd=dest, env=env, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"dbt {command[0]} failed:\n{result.stdout}\n{result.stderr}"
    return dest


@pytest.fixture(scope="module")
def corpus(project: Path) -> list[MetricCase]:
    cases = load_dbt_metrics(project / "target", platform=PLATFORM, cases=FIXTURE / "acme_bench.yml")
    assert not isinstance(cases, DbtError)
    return cases


@pytest.mark.timeout(600)
def test_gold_queries_match_dbt_gold_sql(project: Path, corpus: list[MetricCase]) -> None:
    reference = read_yaml(FIXTURE / "reference_sql.yml", not_found="cases_not_found", invalid="cases_invalid")
    assert isinstance(reference, dict)

    mf_results: dict[str, list[dict[str, str]]] = {}
    for case in corpus:
        rows = run(case.gold, case.target_dir, profiles_dir=case.profiles_dir)
        assert not isinstance(rows, DbtError), f"{case.id} did not run: {rows}"
        mf_results[case.id] = rows

    connection = duckdb.connect(str(project / "acme.duckdb"), read_only=True)
    try:
        connection.execute("set search_path='main'")
        sql_results = {case.id: connection.execute(reference[case.id]).fetchall() for case in corpus}
    finally:
        connection.close()

    for case in corpus:
        got = _mf_index(mf_results[case.id], case.gold.metrics[0])
        want = _sql_index(sql_results[case.id])
        assert got and want, f"{case.id}: empty result (gold query {got}, dbt gold SQL {want})"
        assert got.keys() == want.keys(), f"{case.id}: group keys differ: {sorted(got)} != {sorted(want)}"
        for key, value in got.items():
            assert _close(value, want[key]), f"{case.id} at {key}: gold query {value} != dbt gold SQL {want[key]}"


@pytest.mark.timeout(600)
def test_corpus_scores_pass_through_the_cascade(corpus: list[MetricCase]) -> None:
    scorers = [metric_layer_equivalence(StubLlm(JudgeReply(reason="unused", score=1.0)))]
    summary = run_metric_benchmark(corpus, _GoldSolver(), scorers=scorers)
    assert summary.total == len(corpus)
    assert summary.passed == summary.total
    assert summary.accuracy == 1.0
    # Each gold resolves to itself, so the spec tier decides; the stub judge is never reached.
    assert all(report.scores[0].basis == "proven" for report in summary.cases)
