"""Text-to-SQL evals against a dbt project (jaffle shop), with the model stubbed.

`platform_from_profile` reads the warehouse connection from the project's dbt profile and
`load_dbt` builds one `EvalCase` per question from a cases file. Each case's gold answer is a SQL
query; `ExecutionAccuracy` runs the candidate and the gold and compares their result rows, so a
query written differently from the gold that returns the same rows passes. The project ships a
small DuckDB warehouse, so the file runs offline with no model or network — `StubLlm` supplies the
AI SQL in place of a live model.
"""

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from evaldata import ExecutionAccuracy, PromptSolver, assert_eval
from evaldata.dbt import DbtError, load_dbt, platform_from_profile
from evaldata.llm import StubLlm
from evaldata.platforms.registry import close_all
from evaldata.solvers import SCHEMA_PROMPT_TEMPLATE
from evaldata.types import EvalCase, PlatformRef

_PROJECT = Path(__file__).parent / "jaffle"
_CASES = Path(__file__).parent / "cases.yml"
_SCORER = ExecutionAccuracy(row_order="ignore", multiplicity="set")


@pytest.fixture(autouse=True)
def _close_connections() -> Iterator[None]:
    """Close warehouse connections after each test so a copied DuckDB is never reused across cases."""
    yield
    close_all()


def _cases(tmp_path: Path) -> dict[str, EvalCase]:
    """Copy the project to a temp dir and load its cases, keyed by id.

    Args:
        tmp_path: A temp directory the project is copied into so its DuckDB is never mutated.

    Returns:
        The loaded eval cases, keyed by case id.
    """
    project = tmp_path / "jaffle"
    shutil.copytree(_PROJECT, project)
    platform = platform_from_profile(project)
    assert isinstance(platform, PlatformRef)
    cases = load_dbt(project / "artifacts", platform=platform, cases=_CASES)
    assert not isinstance(cases, DbtError)
    return {case.id: case for case in cases}


def test_reworded_sql_passes(tmp_path: Path) -> None:
    """A query written differently from the gold but returning the same rows passes: results are compared, not SQL text."""
    case = _cases(tmp_path)["customers-count"]
    solver = PromptSolver(
        model=StubLlm("SELECT COUNT(customer_id) AS total FROM stg_customers"),
        prompt_template=SCHEMA_PROMPT_TEMPLATE,
    )
    assert_eval(case, solver, scorers=[_SCORER])


def test_wrong_sql_fails(tmp_path: Path) -> None:
    """A query that returns the wrong rows fails the eval."""
    case = _cases(tmp_path)["customers-count"]
    solver = PromptSolver(model=StubLlm("SELECT 1 AS n"), prompt_template=SCHEMA_PROMPT_TEMPLATE)
    with pytest.raises(AssertionError):
        assert_eval(case, solver, scorers=[_SCORER])


def test_revenue_by_region_passes(tmp_path: Path) -> None:
    """Total revenue by customer region — a join across orders and customers, scored on its result rows."""
    case = _cases(tmp_path)["revenue-by-region"]
    sql = (
        "SELECT c.region, SUM(o.amount) AS revenue "
        "FROM stg_orders o JOIN stg_customers c ON o.customer_id = c.customer_id "
        "GROUP BY c.region"
    )
    solver = PromptSolver(model=StubLlm(sql), prompt_template=SCHEMA_PROMPT_TEMPLATE)
    assert_eval(case, solver, scorers=[_SCORER])
