# Evaluate Snowflake Cortex Analyst

[Cortex Analyst](https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-analyst) turns a
natural-language question into SQL against a semantic model. `CortexAnalystSolver` sends each
case's question to the Cortex Analyst REST endpoint, returns the generated SQL, and `evaldata` runs
that SQL on Snowflake and scores it against your gold query.

## Prerequisites

```bash
uv add "evaldata[cortex]"
```

The Snowflake role used by the connection needs the `SNOWFLAKE.CORTEX_USER` database role (granted
to `PUBLIC` by default) and access to a semantic model to answer against.

## Provision a semantic view

Cortex Analyst answers against a semantic model. This guide uses a
[semantic view](https://docs.snowflake.com/en/user-guide/views-semantic/overview) — a native
object created with DDL. The example is a two-table jaffle shop:

```sql
CREATE DATABASE IF NOT EXISTS JAFFLE_SHOP_DB;

CREATE OR REPLACE TABLE JAFFLE_SHOP_DB.PUBLIC.CUSTOMERS (ID NUMBER, NAME VARCHAR, REGION VARCHAR);
CREATE OR REPLACE TABLE JAFFLE_SHOP_DB.PUBLIC.ORDERS
  (ID NUMBER, CUSTOMER_ID NUMBER, ORDER_DATE DATE, AMOUNT NUMBER(10,2));

INSERT INTO JAFFLE_SHOP_DB.PUBLIC.CUSTOMERS (ID, NAME, REGION) VALUES
  (1, 'Alice', 'East'), (2, 'Bob', 'West'), (3, 'Carol', 'East'), (4, 'Dan', 'West'), (5, 'Eve', 'East');
INSERT INTO JAFFLE_SHOP_DB.PUBLIC.ORDERS (ID, CUSTOMER_ID, ORDER_DATE, AMOUNT) VALUES
  (1, 1, '2026-01-05', 50.00), (2, 1, '2026-01-20', 30.00), (3, 2, '2026-02-01', 120.00),
  (4, 3, '2026-02-15', 75.50), (5, 3, '2026-03-01', 20.00), (6, 4, '2026-03-10', 200.00),
  (7, 5, '2026-03-12', 10.00), (8, 2, '2026-03-15', 60.00);

CREATE OR REPLACE SEMANTIC VIEW JAFFLE_SHOP_DB.PUBLIC.JAFFLE_SHOP_SV
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
  );
```

Cortex Analyst also accepts a semantic-model YAML file on a stage; pass `semantic_model_file` to
the solver instead of `semantic_view` to use one.

## Write the eval

`CortexAnalystClient.from_connection` sends questions using the account host and session token of a
Snowflake connection, so the connection you evaluate against also authenticates the Cortex Analyst
call. Authentication is configured as for any Snowflake connection (see the
[Snowflake guide](snowflake.md#authentication)). Score the generated SQL against a gold query with
`ExecutionAccuracy`, which runs both queries and compares the rows by value:

```python
--8<-- "examples/08_cortex/test_cortex_analyst.py"
```

When Cortex Analyst returns suggestions instead of SQL (an ambiguous question), the solver reports
an `empty_response` `SolverError`.

## Run it

```bash
uv run pytest test_cortex.py -q
```

Each case sends one Cortex Analyst message and runs the SQL on your warehouse, both of which
consume Snowflake credits. See the [Support & cost policy](../support-policy.md) for how evaldata
tests Cortex Analyst without spending credits on every change.

## Accuracy

evaldata includes an off-by-default live benchmark of five questions against this semantic view. It
scored **100% (5/5)** execution accuracy on 2026-07-08. Cortex Analyst generates SQL with a model,
so that number is a dated snapshot — run the benchmark on your own account to reproduce it.

## Next steps

- [Support & cost policy](../support-policy.md) — how live and replayed tests are structured.
- [Evaluate against Snowflake](snowflake.md) — the adapter, authentication, and warehouse pushdown.
- [Cortex reference](../reference/cortex.md) — the solver and client API.
