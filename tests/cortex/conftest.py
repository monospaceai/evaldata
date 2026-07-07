"""Shared fixtures for the Cortex Analyst tests: vcr scrubbing, replay/record client, live e2e.

The vcr config strips the `Authorization` header and rewrites the account host to a placeholder
before a cassette is written, so a recording carries no credential and no account identifier.
Recording (against the live endpoint) is opt-in via the `CORTEX_RECORD` environment variable;
without it the client points at the placeholder host and replays the committed cassette offline.
The `live_adapter`/`jaffle_view` fixtures back the `cortex`-marked live e2e.
"""

import os
from collections.abc import Iterator

import pytest

from evaldata.cortex.client import CortexAnalystClient
from evaldata.platforms.base import PlatformAdapter

PLACEHOLDER_HOST = "test-account.snowflakecomputing.com"

DATABASE = "JAFFLE_SHOP_DB"
SCHEMA = "PUBLIC"
SEMANTIC_VIEW = f"{DATABASE}.{SCHEMA}.JAFFLE_SHOP_SV"

_JAFFLE_DDL = [
    f"CREATE DATABASE IF NOT EXISTS {DATABASE}",
    f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.{SCHEMA}",
    f"CREATE OR REPLACE TABLE {DATABASE}.{SCHEMA}.CUSTOMERS (ID NUMBER, NAME VARCHAR, REGION VARCHAR)",
    f"CREATE OR REPLACE TABLE {DATABASE}.{SCHEMA}.ORDERS "
    "(ID NUMBER, CUSTOMER_ID NUMBER, ORDER_DATE DATE, AMOUNT NUMBER(10,2))",
    f"INSERT INTO {DATABASE}.{SCHEMA}.CUSTOMERS (ID, NAME, REGION) VALUES "
    "(1,'Alice','East'),(2,'Bob','West'),(3,'Carol','East'),(4,'Dan','West'),(5,'Eve','East')",
    f"INSERT INTO {DATABASE}.{SCHEMA}.ORDERS (ID, CUSTOMER_ID, ORDER_DATE, AMOUNT) VALUES "
    "(1,1,'2026-01-05',50.00),(2,1,'2026-01-20',30.00),(3,2,'2026-02-01',120.00),"
    "(4,3,'2026-02-15',75.50),(5,3,'2026-03-01',20.00),(6,4,'2026-03-10',200.00),"
    "(7,5,'2026-03-12',10.00),(8,2,'2026-03-15',60.00)",
    f"""CREATE OR REPLACE SEMANTIC VIEW {SEMANTIC_VIEW}
  TABLES (
    customers AS {DATABASE}.{SCHEMA}.CUSTOMERS PRIMARY KEY (ID) COMMENT = 'One row per customer',
    orders AS {DATABASE}.{SCHEMA}.ORDERS PRIMARY KEY (ID) COMMENT = 'One row per order'
  )
  RELATIONSHIPS (
    orders_to_customers AS orders (CUSTOMER_ID) REFERENCES customers
  )
  FACTS (
    orders.amount AS orders.AMOUNT
  )
  DIMENSIONS (
    customers.customer_name AS customers.NAME WITH SYNONYMS = ('customer') COMMENT = 'Customer name',
    customers.region AS customers.REGION WITH SYNONYMS = ('area') COMMENT = 'Customer region',
    orders.order_date AS orders.ORDER_DATE COMMENT = 'Date the order was placed'
  )
  METRICS (
    orders.order_count AS COUNT(orders.ID) COMMENT = 'Number of orders',
    orders.total_amount AS SUM(orders.AMOUNT) COMMENT = 'Total order amount'
  )
  COMMENT = 'Minimal jaffle-shop model for Cortex Analyst'""",
]


def _connect_snowflake() -> PlatformAdapter:
    """Connect a Snowflake adapter from the `SNOWFLAKE_*` environment (fail-loud)."""
    from evaldata.platforms.snowflake import SnowflakeAdapter

    return SnowflakeAdapter(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ.get("SNOWFLAKE_USER"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
        private_key_file=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"),
        authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR"),
        token=os.environ.get("SNOWFLAKE_TOKEN"),
        private_key_file_pwd=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"),
        workload_identity_provider=os.environ.get("SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"),
    )


def _scrub_request(request: object) -> object:
    """Drop the auth header and rewrite the account host to `PLACEHOLDER_HOST`."""
    import re

    headers = request.headers  # type: ignore[attr-defined]
    for key in list(headers):
        if key.lower() == "authorization":
            del headers[key]
    request.uri = re.sub(r"https://[^/]+/", f"https://{PLACEHOLDER_HOST}/", request.uri)  # type: ignore[attr-defined]
    return request


@pytest.fixture(scope="module")
def vcr_config() -> dict[str, object]:
    """Configure vcrpy to scrub credentials and match without host or body."""
    return {
        "filter_headers": [("authorization", None)],
        "before_record_request": _scrub_request,
        "match_on": ["method", "path"],
    }


@pytest.fixture
def cortex_vcr_client() -> CortexAnalystClient:
    """A live client when recording (`CORTEX_RECORD` set), else a placeholder-host replay client."""
    if os.environ.get("CORTEX_RECORD"):
        return CortexAnalystClient.from_connection(_connect_snowflake().connection)
    return CortexAnalystClient(host=PLACEHOLDER_HOST, token_provider=lambda: "dummy-token")


@pytest.fixture(scope="module")
def live_adapter() -> Iterator[PlatformAdapter]:
    """A live Snowflake adapter for the `cortex`-marked e2e (fail-loud on missing credentials)."""
    adapter = _connect_snowflake()
    yield adapter
    adapter.close()


@pytest.fixture(scope="module")
def jaffle_view(live_adapter: PlatformAdapter) -> str:
    """Build the jaffle-shop tables, seed rows, and semantic view; return the view's name."""
    for statement in _JAFFLE_DDL:
        result = live_adapter.execute(statement)
        if result.error is not None:
            msg = f"jaffle fixture setup failed: {result.error.message}"
            raise RuntimeError(msg)
    return SEMANTIC_VIEW
