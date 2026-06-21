"""Semantic-equivalence example evals: AI SQL that differs syntactically but is semantically equivalent.

`SemanticEquivalence` compares normalized syntax trees first, then falls back to executing
both queries and diffing their result sets.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from evaldata import CallableSolver, EvalCase, SemanticEquivalence, assert_eval, eval_case
from evaldata.platforms import duckdb_platform

_DB_PATH = Path(tempfile.mkdtemp(prefix="evaldata_ex01_sem_")) / "shop.duckdb"
_PLATFORM = duckdb_platform(name="examples-semantic-equivalence", path=str(_DB_PATH))


@pytest.fixture(scope="module", autouse=True)
def _seed_db() -> Iterator[None]:
    con = duckdb.connect(str(_DB_PATH))
    con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, country VARCHAR)")
    con.execute("INSERT INTO customers VALUES (1, 'Ada', 'GB'), (2, 'Bo', 'US'), (3, 'Cy', 'US')")
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL(10, 2))")
    con.execute("INSERT INTO orders VALUES (1, 1, 10.00), (2, 1, 5.50), (3, 2, 20.00), (4, 2, 7.25)")
    con.close()
    yield


# Decided by AstEquivalence: predicate order and casing differ, but the syntax trees match.
@eval_case(
    input="Which US customers have an id above 1?",
    expected={
        "kind": "gold_query",
        "sql": "SELECT name FROM customers WHERE country = 'US' AND id > 1",
    },
    platform=_PLATFORM,
)
def test_confirmed_by_ast(case: EvalCase) -> None:
    """Predicate order and formatting differ; the normalized syntax trees match."""
    solver = CallableSolver(lambda c: "select NAME from customers where id > 1 and country = 'US'")
    assert_eval(case, solver, scorers=[SemanticEquivalence()])


# AstEquivalence abstains on commutative arithmetic (`1 + amount` vs `amount + 1`); execution decides.
@eval_case(
    input="For each order, show its id and amount plus one.",
    expected={
        "kind": "gold_query",
        "sql": "SELECT id, amount + 1 AS bumped FROM orders",
    },
    platform=_PLATFORM,
)
def test_confirmed_by_execution(case: EvalCase) -> None:
    """Commutative arithmetic the syntax check misses, confirmed by running the queries."""
    solver = CallableSolver(lambda c: "SELECT id, 1 + amount AS bumped FROM orders")
    assert_eval(case, solver, scorers=[SemanticEquivalence()])
