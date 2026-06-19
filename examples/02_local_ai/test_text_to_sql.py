"""Local-AI text-to-SQL example evals: a `PromptSolver` calling a local Ollama model."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from dataeval import EvalCase, ResultSetEquivalence, assert_eval, eval_case
from dataeval.platforms import duckdb_platform
from dataeval.solvers import PromptSolver

_DB_PATH = Path(tempfile.mkdtemp(prefix="dataeval_ex02_")) / "shop.duckdb"
_PLATFORM = duckdb_platform(name="examples-local-ai", path=str(_DB_PATH))
_MODEL = os.environ.get("DATAEVAL_LOCAL_MODEL", "")


@pytest.fixture(scope="module", autouse=True)
def _require_local_model() -> None:
    # Fail loudly (never skip) when these examples run without a configured local model.
    if not _MODEL:  # pragma: no cover
        msg = "set DATAEVAL_LOCAL_MODEL to your local model's id, e.g. ollama_chat/qwen2.5-coder:1.5b"
        raise RuntimeError(msg)


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
    input="Return the id column of every row in the orders table, one row per order.",
    expected={"rows": [{"id": 1}, {"id": 2}, {"id": 3}, {"id": 4}]},
    platform=_PLATFORM,
)
def test_select_order_ids(case: EvalCase) -> None:
    """Local model selects every order id; scored on exact rows."""
    solver = PromptSolver(model=_MODEL, timeout=120, temperature=0)
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


@eval_case(
    input="Return the name column of every customer whose country is 'US', one row per customer.",
    expected={"rows": [{"name": "Bo"}, {"name": "Cy"}]},
    platform=_PLATFORM,
)
def test_select_us_customer_names(case: EvalCase) -> None:
    """Local model selects US customer names; scored on exact rows."""
    solver = PromptSolver(model=_MODEL, timeout=120, temperature=0)
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])
