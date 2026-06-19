"""Databricks platform example evals: fixed-SQL `CallableSolver` against a live SQL Warehouse.

Where examples 01-03 vary the *solver* against a local DuckDB, this varies the *platform*:
the same kinds of cases run against a real Databricks SQL Warehouse. The solver is
deterministic so the focus is the platform — precise type resolution, warehouse pushdown,
and secretless unified auth.

Connection details come from `DATABRICKS_SERVER_HOSTNAME` / `DATABRICKS_HTTP_PATH`;
credentials resolve from the ambient environment via the Databricks SDK (nothing secret is
held in the `PlatformRef`). Every case is marked `cloud`: the default `check` runs them; the
`check-nocloud` fast path skips them, and a connection is opened only when a case runs.
"""

import os
from collections.abc import Iterator
from decimal import Decimal

import pytest

from dataeval import (
    CallableSolver,
    EvalCase,
    ExpectationSuiteScorer,
    ResultSetEquivalence,
    assert_eval,
    eval_case,
)
from dataeval.platforms import databricks_platform, resolve

pytestmark = [pytest.mark.e2e, pytest.mark.cloud]

_VIEW = "dataeval_ex04_orders"
_PLATFORM = databricks_platform(
    name="examples-databricks",
    server_hostname=os.environ.get("DATABRICKS_SERVER_HOSTNAME", ""),
    http_path=os.environ.get("DATABRICKS_HTTP_PATH", ""),
)


@pytest.fixture(scope="module", autouse=True)
def _seed_warehouse() -> Iterator[None]:
    # A session-scoped TEMPORARY VIEW: visible only on this connection, gone when it closes —
    # no catalog to choose, no table to clean up, only query permissions needed. Seeded on the
    # same cached adapter the cases resolve, so the view (and the DESCRIBE QUERY type probe)
    # see it. Explicit CASTs pin the column types the typed case asserts.
    adapter = resolve(_PLATFORM)
    result = adapter.execute(
        f"CREATE OR REPLACE TEMPORARY VIEW {_VIEW} AS "
        "SELECT CAST(id AS INT) AS id, CAST(customer AS STRING) AS customer, "
        "CAST(amount AS DECIMAL(10, 2)) AS amount "
        "FROM VALUES (1, 'Ada', 10.00), (2, 'Bo', 5.50), (3, 'Cy', 20.00) AS t(id, customer, amount)"
    )
    if result.error is not None:
        msg = f"failed to seed Databricks temp view {_VIEW!r}: {result.error}"
        raise RuntimeError(msg)
    yield


# Typed result set with precise types. The `DECIMAL(10, 2)` assertion passes only because
# dataeval recovers the scale via `DESCRIBE QUERY`: the connector's raw column description
# reports a bare `DECIMAL` (i.e. `DECIMAL(10, 0)`), which would fail this assertion.
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


# Untyped result set: values only. No column-type assertion, so it sidesteps Spark's
# decimal-sum type promotion and checks the computed value in the warehouse.
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
# accuracy). The solver phrases its SQL differently but yields the same rows, so it passes —
# both run in the warehouse and the comparison is on results, not SQL text.
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


# Expectation suite: structural assertions pushed into the warehouse as SQL
# (`row_count` / `not_null` / `unique`) instead of pulling rows back to compare.
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
