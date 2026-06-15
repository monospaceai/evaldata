"""Hosted-AI text-to-SQL example evals: a `PromptSolver` calling a hosted model."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from dataeval import EvalCase, ResultSetEquivalence, assert_eval, eval_case
from dataeval.platforms import duckdb_platform
from dataeval.solvers import PromptSolver

_DB_PATH = Path(tempfile.mkdtemp(prefix="dataeval_ex03_")) / "shop.duckdb"
_PLATFORM = duckdb_platform(name="examples-hosted-ai", path=str(_DB_PATH))
_MODEL = os.getenv("DATA_EVAL_HOSTED_MODEL", "openai/gpt-4o-mini")


@pytest.fixture(scope="module", autouse=True)
def _seed_db() -> Iterator[None]:
    con = duckdb.connect(str(_DB_PATH))
    con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, country VARCHAR)")
    con.execute("INSERT INTO customers VALUES (1, 'Ada', 'GB'), (2, 'Bo', 'US'), (3, 'Cy', 'US')")
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL(10, 2))")
    con.execute("INSERT INTO orders VALUES (1, 1, 10.00), (2, 1, 5.50), (3, 2, 20.00), (4, 2, 7.25)")
    con.close()
    yield


@eval_case(
    input="How many orders are there? Name the output column order_count.",
    expected={"rows": [{"order_count": 4}]},
    platform=_PLATFORM,
)
def test_order_count(case: EvalCase) -> None:
    """Hosted model counts orders; scored on exact rows."""
    solver = PromptSolver(model=_MODEL, timeout=120, temperature=0)
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


@eval_case(
    input=("How many distinct customers placed an order? Name the output column customer_count."),
    expected={"rows": [{"customer_count": 2}]},
    platform=_PLATFORM,
)
def test_distinct_customers(case: EvalCase) -> None:
    """Hosted model counts distinct ordering customers; scored on exact rows."""
    solver = PromptSolver(model=_MODEL, timeout=120, temperature=0)
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])
