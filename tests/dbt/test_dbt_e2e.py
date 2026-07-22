"""End-to-end dbt eval against a real DuckDB warehouse, with the LLM stubbed."""

import shutil
from pathlib import Path

import pytest

from evaldata.core import run_benchmark
from evaldata.dbt import DbtError, load_dbt, platform_from_profile
from evaldata.llm import StubLlm
from evaldata.platforms.registry import close_all
from evaldata.scorers import ExecutionAccuracy
from evaldata.solvers import SCHEMA_PROMPT_TEMPLATE, PromptSolver
from evaldata.types import DuckDBPlatformRef

pytestmark = pytest.mark.e2e

FIXTURE = Path(__file__).parent / "fixtures" / "jaffle_duckdb"
_CASES = "- question: How many customers are there?\n  gold_sql: select count(*) as n from customers\n"


def _project(tmp_path: Path) -> Path:
    # Copy the project so executing against its DuckDB never touches the committed fixture.
    dest = tmp_path / "jaffle"
    shutil.copytree(FIXTURE, dest, ignore=shutil.ignore_patterns("target", "logs", "dbt_packages"))
    (dest / "cases.yml").write_text(_CASES, encoding="utf-8")
    return dest


def _summary(tmp_path: Path, model_sql: str) -> float:
    project = _project(tmp_path)
    platform = platform_from_profile(project)
    assert isinstance(platform, DuckDBPlatformRef)
    cases = load_dbt(project / "artifacts", platform=platform, cases=project / "cases.yml")
    assert not isinstance(cases, DbtError)
    solver = PromptSolver(model=StubLlm(model_sql), prompt_template=SCHEMA_PROMPT_TEMPLATE)
    try:
        return run_benchmark(
            cases, solver, scorers=[ExecutionAccuracy(row_order="ignore", multiplicity="set")]
        ).accuracy
    finally:
        close_all()


def test_correct_sql_passes(tmp_path: Path) -> None:
    assert _summary(tmp_path, "select count(*) as n from customers") == 1.0


def test_wrong_sql_fails(tmp_path: Path) -> None:
    assert _summary(tmp_path, "select 1 as n") == 0.0
