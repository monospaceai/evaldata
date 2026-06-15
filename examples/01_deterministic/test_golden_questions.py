"""Deterministic text-to-SQL example evals: a `CallableSolver` returning fixed SQL."""

import tempfile
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from dataeval import (
    CallableSolver,
    EvalCase,
    ExpectationSuiteScorer,
    ResultSetEquivalence,
    assert_eval,
    eval_case,
)
from dataeval.platforms import duckdb_platform

_DB_PATH = Path(tempfile.mkdtemp(prefix="dataeval_ex01_")) / "shop.duckdb"
_PLATFORM = duckdb_platform(name="examples-deterministic", path=str(_DB_PATH))


@pytest.fixture(scope="module", autouse=True)
def _seed_db() -> Iterator[None]:
    con = duckdb.connect(str(_DB_PATH))
    con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, country VARCHAR)")
    con.execute("INSERT INTO customers VALUES (1, 'Ada', 'GB'), (2, 'Bo', 'US'), (3, 'Cy', 'US')")
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL(10, 2))")
    con.execute("INSERT INTO orders VALUES (1, 1, 10.00), (2, 1, 5.50), (3, 2, 20.00), (4, 2, 7.25)")
    con.close()
    yield


# Untyped result set: compare values only (no column types asserted).
@eval_case(
    input="What is the total order amount?",
    expected={"rows": [{"total": Decimal("42.75")}]},
    platform=_PLATFORM,
)
def test_untyped_result_set(case: EvalCase) -> None:
    """Compare result values only, asserting no column types."""
    solver = CallableSolver(lambda c: "SELECT sum(amount) AS total FROM orders")
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Typed result set: same value as above, plus a column-type assertion. This variant also
# fails if the model returns the right number with the wrong type (e.g. DOUBLE or VARCHAR),
# which the untyped variant accepts.
@eval_case(
    input="What is the total order amount?",
    expected={
        "rows": [{"total": Decimal("42.75")}],
        "schema": [{"name": "total", "type": "DECIMAL(38, 2)"}],
    },
    platform=_PLATFORM,
)
def test_typed_result_set(case: EvalCase) -> None:
    """Compare result values plus a column-type assertion."""
    solver = CallableSolver(lambda c: "SELECT sum(amount) AS total FROM orders")
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Gold query: the reference query's executed RESULT is the expected answer (execution
# accuracy). The solver's SQL is phrased differently but yields the same rows, so it
# passes — the comparison is on results, not on SQL text.
@eval_case(
    input="What is the total order amount per customer?",
    expected={
        "kind": "gold_query",
        "sql": ("SELECT customer_id, sum(amount) AS total FROM orders GROUP BY customer_id"),
    },
    platform=_PLATFORM,
)
def test_gold_query(case: EvalCase) -> None:
    """Score against a reference query's executed result (execution accuracy)."""
    solver = CallableSolver(
        lambda c: "SELECT customer_id, sum(amount) AS total FROM orders GROUP BY 1 ORDER BY customer_id DESC"
    )
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Expectation suite: assert structural properties of the result instead of exact rows.
@eval_case(
    input="List every customer with their id and name.",
    expected={
        "kind": "expectation_suite",
        "expectations": [
            {"kind": "row_count", "exact": 3},
            {"kind": "not_null", "column": "id"},
            {"kind": "unique", "column": "id"},
        ],
    },
    platform=_PLATFORM,
)
def test_expectation_suite(case: EvalCase) -> None:
    """Assert structural properties of the result instead of exact rows."""
    solver = CallableSolver(lambda c: "SELECT id, name FROM customers")
    assert_eval(case, solver, scorers=[ExpectationSuiteScorer()])
