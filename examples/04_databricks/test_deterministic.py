"""Databricks example evals: fixed SQL run against a live Databricks SQL Warehouse.

Precise column types are resolved from the warehouse, and expectation and equivalence checks
are pushed down into SQL.

Point it at a warehouse with `DATABRICKS_SERVER_HOSTNAME` and `DATABRICKS_HTTP_PATH`.
Credentials are not part of the platform reference — authenticate with the Databricks CLI
(https://github.com/databricks/cli).
"""

import os
from collections.abc import Iterator
from decimal import Decimal

import pytest

from evaldata import (
    CallableSolver,
    EvalCase,
    ExpectationSuiteScorer,
    ResultSetEquivalence,
    assert_eval,
    eval_case,
)
from evaldata.platforms import databricks_platform, resolve

pytestmark = [pytest.mark.e2e, pytest.mark.cloud]

_VIEW = "evaldata_ex04_orders"
_PLATFORM = databricks_platform(
    name="examples-databricks",
    server_hostname=os.environ.get("DATABRICKS_SERVER_HOSTNAME", ""),
    http_path=os.environ.get("DATABRICKS_HTTP_PATH", ""),
)


@pytest.fixture(scope="module", autouse=True)
def _seed_warehouse() -> Iterator[None]:
    # A session-scoped temp view: nothing lands in the catalog and there's nothing to clean
    # up, so this needs only query permissions. The CASTs fix the types the typed case asserts.
    adapter = resolve(_PLATFORM)
    result = adapter.execute(
        f"CREATE OR REPLACE TEMPORARY VIEW {_VIEW} AS "
        "SELECT CAST(id AS INT) AS id, CAST(customer AS STRING) AS customer, "
        "CAST(amount AS DECIMAL(10, 2)) AS amount "
        "FROM VALUES (1, 'Ada', 10.00), (2, 'Bo', 5.50), (3, 'Cy', 20.00) AS t(id, customer, amount)"
    )
    if result.error is not None:  # pragma: no cover
        msg = f"failed to seed Databricks temp view {_VIEW!r}: {result.error}"
        raise RuntimeError(msg)
    yield


# Precise column types. The `DECIMAL(10, 2)` assertion holds only because evaldata resolves
# the column types from the warehouse — the driver's own description reports a scaleless `DECIMAL`.
@eval_case(
    input="List each order's customer and amount, ordered by id.",
    expected={
        "rows": [
            {"customer": "Ada", "amount": Decimal("10.00")},
            {"customer": "Bo", "amount": Decimal("5.50")},
            {"customer": "Cy", "amount": Decimal("20.00")},
        ],
        "schema": [
            {"name": "customer", "type": "STRING"},
            {"name": "amount", "type": "DECIMAL(10, 2)"},
        ],
    },
    platform=_PLATFORM,
)
def test_precise_types_resolved(case: EvalCase) -> None:
    """Assert exact rows plus precise column types recovered from the warehouse."""
    solver = CallableSolver(lambda c: f"SELECT customer, amount FROM {_VIEW} ORDER BY id")
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Untyped result set: values only, no column-type assertion.
@eval_case(
    input="What is the total order amount?",
    expected={"rows": [{"total": Decimal("35.50")}]},
    platform=_PLATFORM,
)
def test_untyped_total(case: EvalCase) -> None:
    """Compare a warehouse-computed aggregate by value only."""
    solver = CallableSolver(lambda c: f"SELECT sum(amount) AS total FROM {_VIEW}")
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Gold query: the reference query's executed RESULT is the expected answer (execution
# accuracy). The comparison is on the executed result, not the SQL text, so any query that
# returns the same rows passes.
@eval_case(
    input="What is the total order amount per customer?",
    expected={
        "kind": "gold_query",
        "sql": f"SELECT customer, sum(amount) AS total FROM {_VIEW} GROUP BY customer",
    },
    platform=_PLATFORM,
)
def test_gold_query(case: EvalCase) -> None:
    """Score against a reference query's executed result (execution accuracy)."""
    solver = CallableSolver(
        lambda c: f"SELECT customer, sum(amount) AS total FROM {_VIEW} GROUP BY 1 ORDER BY customer DESC"
    )
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Expectation suite: structural assertions (`row_count` / `not_null` / `unique`) pushed into
# the warehouse and evaluated as server-side SQL.
@eval_case(
    input="List every order's id and customer.",
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
def test_expectation_suite_pushdown(case: EvalCase) -> None:
    """Assert structural properties of the result, evaluated server-side."""
    solver = CallableSolver(lambda c: f"SELECT id, customer FROM {_VIEW}")
    assert_eval(case, solver, scorers=[ExpectationSuiteScorer()])
