"""Cortex Analyst example eval: a live question answered by Snowflake Cortex Analyst.

`CortexAnalystSolver` sends the question to the Cortex Analyst REST endpoint and returns the SQL it
generates; `evaldata` runs that SQL on Snowflake and scores it against a gold query.

Set `SNOWFLAKE_ACCOUNT` (and `SNOWFLAKE_WAREHOUSE`/`SNOWFLAKE_ROLE`) and configure authentication
through the environment (see the Snowflake guide). The account needs the `SNOWFLAKE.CORTEX_USER`
database role.
"""

import os
from collections.abc import Iterator
from typing import cast

import pytest

from evaldata import EvalCase, ExecutionAccuracy, assert_eval, eval_case
from evaldata.cortex import CortexAnalystClient, CortexAnalystSolver
from evaldata.platforms import resolve, snowflake_platform
from evaldata.platforms.snowflake import SnowflakeAdapter

pytestmark = [pytest.mark.e2e, pytest.mark.cortex]

_SEMANTIC_VIEW = "JAFFLE_SHOP_DB.PUBLIC.JAFFLE_SHOP_SV"
_PLATFORM = snowflake_platform(
    name="examples-cortex",
    account=os.environ.get("SNOWFLAKE_ACCOUNT", ""),
    user=os.environ.get("SNOWFLAKE_USER"),
    warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
    role=os.environ.get("SNOWFLAKE_ROLE"),
    authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR"),
    workload_identity_provider=os.environ.get("SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"),
)

_SETUP = [
    "CREATE DATABASE IF NOT EXISTS JAFFLE_SHOP_DB",
    "CREATE OR REPLACE TABLE JAFFLE_SHOP_DB.PUBLIC.CUSTOMERS (ID NUMBER, NAME VARCHAR, REGION VARCHAR)",
    "CREATE OR REPLACE TABLE JAFFLE_SHOP_DB.PUBLIC.ORDERS "
    "(ID NUMBER, CUSTOMER_ID NUMBER, ORDER_DATE DATE, AMOUNT NUMBER(10,2))",
    "INSERT INTO JAFFLE_SHOP_DB.PUBLIC.CUSTOMERS (ID, NAME, REGION) VALUES "
    "(1,'Alice','East'),(2,'Bob','West'),(3,'Carol','East'),(4,'Dan','West'),(5,'Eve','East')",
    "INSERT INTO JAFFLE_SHOP_DB.PUBLIC.ORDERS (ID, CUSTOMER_ID, ORDER_DATE, AMOUNT) VALUES "
    "(1,1,'2026-01-05',50.00),(2,1,'2026-01-20',30.00),(3,2,'2026-02-01',120.00),"
    "(4,3,'2026-02-15',75.50),(5,3,'2026-03-01',20.00),(6,4,'2026-03-10',200.00),"
    "(7,5,'2026-03-12',10.00),(8,2,'2026-03-15',60.00)",
    f"""CREATE OR REPLACE SEMANTIC VIEW {_SEMANTIC_VIEW}
  TABLES (
    customers AS JAFFLE_SHOP_DB.PUBLIC.CUSTOMERS PRIMARY KEY (ID) COMMENT = 'One row per customer',
    orders AS JAFFLE_SHOP_DB.PUBLIC.ORDERS PRIMARY KEY (ID) COMMENT = 'One row per order'
  )
  RELATIONSHIPS (
    orders_to_customers AS orders (CUSTOMER_ID) REFERENCES customers
  )
  FACTS (
    orders.amount AS orders.AMOUNT
  )
  DIMENSIONS (
    customers.customer_name AS customers.NAME COMMENT = 'Customer name',
    customers.region AS customers.REGION COMMENT = 'Customer region',
    orders.order_date AS orders.ORDER_DATE COMMENT = 'Date the order was placed'
  )
  METRICS (
    orders.order_count AS COUNT(orders.ID) COMMENT = 'Number of orders',
    orders.total_amount AS SUM(orders.AMOUNT) COMMENT = 'Total order amount'
  )""",
]


@pytest.fixture(scope="module", autouse=True)
def _seed_semantic_view() -> Iterator[None]:
    adapter = resolve(_PLATFORM)
    for sql in _SETUP:
        result = adapter.execute(sql)
        if result.error is not None:  # pragma: no cover
            msg = f"failed to build the jaffle semantic view: {result.error.message}"
            raise RuntimeError(msg)
    yield


@eval_case(
    input="What is the total order amount for each customer region?",
    expected={
        "kind": "gold_query",
        "sql": "SELECT c.REGION, SUM(o.AMOUNT) FROM JAFFLE_SHOP_DB.PUBLIC.ORDERS o "
        "JOIN JAFFLE_SHOP_DB.PUBLIC.CUSTOMERS c ON o.CUSTOMER_ID = c.ID GROUP BY c.REGION",
    },
    platform=_PLATFORM,
)
def test_cortex_answers_region_totals(case: EvalCase) -> None:
    """Ask Cortex Analyst the question and score its SQL against the gold query's rows."""
    adapter = cast(SnowflakeAdapter, resolve(_PLATFORM))
    solver = CortexAnalystSolver(CortexAnalystClient.from_connection(adapter.connection), semantic_view=_SEMANTIC_VIEW)
    assert_eval(case, solver, scorers=[ExecutionAccuracy(row_order="ignore", column_alignment="by_value")])
