"""BigQuery example evals: fixed SQL run against a live BigQuery project.

The examples check result-column types, run expectations in BigQuery, and compare one result with
a gold query.

Set `BIGQUERY_PROJECT` and configure Application Default Credentials (see the BigQuery guide).
Seeding targets `BIGQUERY_DATASET` when set, defaulting to `evaldata_examples`.
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
from evaldata.platforms import bigquery_platform, resolve
from evaldata.types import ExecutionFailure

pytestmark = [pytest.mark.e2e, pytest.mark.cloud, pytest.mark.bigquery]

_PROJECT = os.environ.get("BIGQUERY_PROJECT", "your-project-id")
_DATASET = os.environ.get("BIGQUERY_DATASET", "evaldata_examples")
_TABLE = f"{_PROJECT}.{_DATASET}.orders_ex09"
_PLATFORM = bigquery_platform(
    name="examples-bigquery",
    project=_PROJECT,
    dataset=_DATASET,
    location=os.environ.get("BIGQUERY_LOCATION"),
)


@pytest.fixture(scope="module", autouse=True)
def _seed_project() -> Iterator[None]:
    adapter = resolve(_PLATFORM)
    statements = []
    if "BIGQUERY_DATASET" not in os.environ:
        statements.append(f"CREATE SCHEMA IF NOT EXISTS `{_PROJECT}.{_DATASET}`")
    statements += [
        f"CREATE OR REPLACE TABLE `{_TABLE}` (id INT64, customer STRING, amount NUMERIC(10, 2))",
        f"INSERT INTO `{_TABLE}` (id, customer, amount) VALUES (1, 'Ada', 10.00), (2, 'Bo', 5.50), (3, 'Cy', 20.00)",
    ]
    for sql in statements:
        result = adapter.execute(sql)
        if isinstance(result, ExecutionFailure):  # pragma: no cover
            msg = f"failed to seed BigQuery table {_TABLE!r}: {result.error.message}"
            raise RuntimeError(msg)
    yield
    adapter.execute(f"DROP TABLE IF EXISTS `{_TABLE}`")


# Column types resolved from the query result: BigQuery reports the stored `NUMERIC(10, 2)`
# column as a bare `NUMERIC` in a query result, since scale is a write-time column constraint.
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
            {"name": "amount", "type": "NUMERIC"},
        ],
    },
    platform=_PLATFORM,
)
def test_precise_types_resolved(case: EvalCase) -> None:
    """Assert exact rows plus precise column types recovered from the query result."""
    solver = CallableSolver(lambda c: f"SELECT customer, amount FROM `{_TABLE}` ORDER BY id")
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Gold query: the reference query's executed RESULT is the expected answer (execution accuracy).
@eval_case(
    input="What is the total order amount per customer?",
    expected={
        "kind": "gold_query",
        "sql": f"SELECT customer, sum(amount) AS total FROM `{_TABLE}` GROUP BY customer",
    },
    platform=_PLATFORM,
)
def test_gold_query(case: EvalCase) -> None:
    """Score an unordered aggregate against a gold query, comparing rows by value."""
    solver = CallableSolver(
        lambda c: f"SELECT customer, sum(amount) AS total FROM `{_TABLE}` GROUP BY 1 ORDER BY customer DESC"
    )
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])


# Expectation suite: structural assertions plus a precise column type, evaluated server-side.
@eval_case(
    input="List every order's id, customer, and amount.",
    expected={
        "kind": "expectation_suite",
        "expectations": [
            {"kind": "row_count", "exact": 3},
            {"kind": "not_null", "column": "id"},
            {"kind": "unique", "column": "id"},
            {"kind": "column_type", "column": "amount", "expected_type": "NUMERIC"},
        ],
    },
    platform=_PLATFORM,
)
def test_expectation_suite_pushdown(case: EvalCase) -> None:
    """Assert structural properties and a precise column type, evaluated server-side."""
    solver = CallableSolver(lambda c: f"SELECT id, customer, amount FROM `{_TABLE}`")
    assert_eval(case, solver, scorers=[ExpectationSuiteScorer()])
