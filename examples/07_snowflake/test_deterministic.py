"""Snowflake example evals: fixed SQL run against a live Snowflake warehouse.

Expectation checks are pushed down into SQL, precise result-column types are resolved from the
warehouse, and execution accuracy compares against a gold query.

Set `SNOWFLAKE_ACCOUNT` (and `SNOWFLAKE_WAREHOUSE`/`SNOWFLAKE_ROLE`) and configure authentication
through the environment (see the Snowflake guide). Seeding targets `SNOWFLAKE_DATABASE` and
`SNOWFLAKE_SCHEMA` when set, defaulting to `EVALDATA_EXAMPLES`/`PUBLIC`.
"""

import os
from collections.abc import Iterator

import pytest

from evaldata import (
    CallableSolver,
    EvalCase,
    ExecutionAccuracy,
    ExpectationSuiteScorer,
    assert_eval,
    eval_case,
)
from evaldata.platforms import resolve, snowflake_platform
from evaldata.types import ExecutionFailure

pytestmark = [pytest.mark.e2e, pytest.mark.cloud, pytest.mark.snowflake]

_DATABASE = os.environ.get("SNOWFLAKE_DATABASE", "EVALDATA_EXAMPLES")
_SCHEMA = os.environ.get("SNOWFLAKE_SCHEMA", "PUBLIC")
_TABLE = f"{_DATABASE}.{_SCHEMA}.ORDERS_EX07"
_PLATFORM = snowflake_platform(
    name="examples-snowflake",
    account=os.environ.get("SNOWFLAKE_ACCOUNT", ""),
    user=os.environ.get("SNOWFLAKE_USER"),
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
    role=os.environ.get("SNOWFLAKE_ROLE"),
    authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR"),
    workload_identity_provider=os.environ.get("SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"),
)


@pytest.fixture(scope="module", autouse=True)
def _seed_warehouse() -> Iterator[None]:
    adapter = resolve(_PLATFORM)
    statements = []
    if "SNOWFLAKE_DATABASE" not in os.environ:
        statements.append(f"CREATE DATABASE IF NOT EXISTS {_DATABASE}")
    if "SNOWFLAKE_SCHEMA" not in os.environ:
        statements.append(f"CREATE SCHEMA IF NOT EXISTS {_DATABASE}.{_SCHEMA}")
    statements += [
        f"CREATE OR REPLACE TABLE {_TABLE} (ID INT, CUSTOMER STRING, AMOUNT DECIMAL(10, 2))",
        f"INSERT INTO {_TABLE} VALUES (1, 'Ada', 10.00), (2, 'Bo', 5.50), (3, 'Cy', 20.00)",
    ]
    for sql in statements:
        result = adapter.execute(sql)
        if isinstance(result, ExecutionFailure):  # pragma: no cover
            msg = f"failed to seed Snowflake table {_TABLE!r}: {result.error.message}"
            raise RuntimeError(msg)
    yield


@eval_case(
    input="List each order's customer and amount, ordered by id.",
    expected={
        "kind": "gold_query",
        "sql": f"SELECT CUSTOMER, AMOUNT FROM {_TABLE} ORDER BY ID",
    },
    platform=_PLATFORM,
)
def test_ordered_list(case: EvalCase) -> None:
    """Score against a reference query's executed rows (execution accuracy)."""
    solver = CallableSolver(lambda c: f"SELECT CUSTOMER, AMOUNT FROM {_TABLE} ORDER BY ID")
    assert_eval(case, solver, scorers=[ExecutionAccuracy()])


@eval_case(
    input="What is the total order amount per customer?",
    expected={
        "kind": "gold_query",
        "sql": f"SELECT CUSTOMER, SUM(AMOUNT) AS TOTAL FROM {_TABLE} GROUP BY CUSTOMER",
    },
    platform=_PLATFORM,
)
def test_gold_query(case: EvalCase) -> None:
    """Score an unordered aggregate against a gold query, comparing rows by value."""
    solver = CallableSolver(
        lambda c: f"SELECT CUSTOMER, SUM(AMOUNT) AS TOTAL FROM {_TABLE} GROUP BY 1 ORDER BY CUSTOMER DESC"
    )
    assert_eval(case, solver, scorers=[ExecutionAccuracy(row_order="ignore")])


# `column_type` resolves the column's precise type from the warehouse: Snowflake reports the
# `AMOUNT` column as `NUMBER(10, 2)`, which matches the authored `DECIMAL(10, 2)`.
@eval_case(
    input="List every order's id, customer, and amount.",
    expected={
        "kind": "expectation_suite",
        "expectations": [
            {"kind": "row_count", "exact": 3},
            {"kind": "not_null", "column": "ID"},
            {"kind": "unique", "column": "ID"},
            {"kind": "column_type", "column": "AMOUNT", "expected_type": "DECIMAL(10, 2)"},
        ],
    },
    platform=_PLATFORM,
)
def test_expectation_suite_pushdown(case: EvalCase) -> None:
    """Assert structural properties and a precise column type, evaluated server-side."""
    solver = CallableSolver(lambda c: f"SELECT ID, CUSTOMER, AMOUNT FROM {_TABLE}")
    assert_eval(case, solver, scorers=[ExpectationSuiteScorer()])
