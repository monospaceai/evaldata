"""End-to-end reproduction of dbt's ACME Semantic Layer benchmark on local DuckDB.

Builds the project from its seeds in a temp dir, then for each of dbt's 11 questions runs the gold
MetricFlow query through `mf` and asserts its rows match dbt's gold SQL on the same warehouse, so
the local port reproduces dbt's benchmark row for row. Also scores the whole corpus through the
cascade.
"""

import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import duckdb
import pytest

from evaldata.dbt import (
    DbtError,
    MetricCase,
    MetricSolverOutput,
    load_dbt_metrics,
    metric_layer_equivalence,
    run,
    run_metric_benchmark,
)
from evaldata.dbt._yaml import read_yaml
from evaldata.llm import StubLlm
from evaldata.scorers.llm_judge import JudgeReply
from evaldata.types import PlatformRef

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "acme_insurance"
PLATFORM = PlatformRef(name="acme", kind="duckdb")


class _GoldSolver:
    """A `MetricSolver` that answers each case with its own gold query."""

    def solve(self, case: MetricCase) -> MetricSolverOutput:
        return MetricSolverOutput(query=case.gold)


def _number(value: object) -> float | None:
    try:
        return round(float(value), 3)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _row_key(measure: object, others: list[object]) -> tuple[float | None, tuple[str, ...]]:
    return _number(measure), tuple(sorted(str(other) for other in others))


def _mf_rows(rows: list[dict[str, str]], metric: str) -> Counter[tuple[float | None, tuple[str, ...]]]:
    return Counter(_row_key(row.get(metric), [v for k, v in row.items() if k != metric]) for row in rows)


def _sql_rows(rows: list[tuple[object, ...]]) -> Counter[tuple[float | None, tuple[str, ...]]]:
    return Counter(_row_key(row[-1], list(row[:-1])) for row in rows)


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
        got = _mf_rows(mf_results[case.id], case.gold.metrics[0])
        want = _sql_rows(sql_results[case.id])
        assert got == want, f"{case.id}: gold query rows {got} != dbt gold SQL rows {want}"


@pytest.mark.timeout(600)
def test_corpus_scores_pass_through_the_cascade(corpus: list[MetricCase]) -> None:
    scorers = [metric_layer_equivalence(StubLlm(JudgeReply(reason="unused", score=1.0)))]
    summary = run_metric_benchmark(corpus, _GoldSolver(), scorers=scorers)
    assert summary.total == len(corpus)
    assert summary.passed == summary.total
    assert summary.accuracy == 1.0
