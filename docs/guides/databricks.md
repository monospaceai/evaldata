# Evaluate text-to-SQL on Databricks

This guide shows how to connect, authenticate, and run SQL evals in a Databricks SQL warehouse.

## Prerequisites

```bash
uv add "evaldata[databricks]"
```

## What this guide covers

- **Type resolution.** The typed case checks `amount: DECIMAL(10, 2)`. `evaldata` gets the
  precision and scale from the warehouse; the driver reports only `DECIMAL`.
- **Warehouse checks.** The `ExpectationSuite` (`row_count` / `not_null` / `unique`) and
  result-set equivalence run as SQL in Databricks.
- **Authentication.** The platform stores the workspace host and HTTP path. The Databricks SDK
  authenticates, for example through `databricks auth login`.

The fixture creates a session-scoped `TEMPORARY VIEW`. It needs only query permissions and leaves
no catalog objects.

## Write the eval

Create `test_databricks.py`:

```python
--8<-- "examples/04_databricks/test_deterministic.py"
```

The example reads its warehouse connection from the environment, so set these before running:

- `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH`: your warehouse's host and HTTP path.
  These are arguments to `databricks_platform()`; pass them as literals if you prefer.
- `DATABRICKS_TOKEN`: read by the Databricks SDK to authenticate. Or use another method it
  supports, e.g. `databricks auth login` for OAuth.

## Run it

```bash
uv run pytest test_databricks.py -q
```

!!! tip "Run it from a clone"
    This is the bundled `examples/04_databricks/` example. If you've cloned the repo, run it
    with `uv run pytest examples/04_databricks`.

## Next steps

- [Concepts](../concepts.md): platforms, scorers, and expected types in depth.
- [Platforms reference](../reference/platforms.md): the adapter API.
