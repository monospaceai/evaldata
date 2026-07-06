# Evaluate against Databricks

Run an eval against a real Databricks SQL warehouse. The solver returns fixed SQL, so the focus
is the platform: `evaldata` resolves precise column types from the warehouse and pushes the
checks down into server-side SQL.

## Prerequisites

```bash
uv add "evaldata[databricks]"
```

## What it demonstrates

- **Precise type resolution** — the typed case asserts `amount: DECIMAL(10, 2)`, which holds
  only because `evaldata` resolves precise column types from the warehouse; the raw driver
  reports a scaleless `DECIMAL`.
- **Warehouse pushdown** — the `ExpectationSuite` (`row_count` / `not_null` / `unique`) and
  result-set equivalence run as SQL server-side, not by pulling rows back to compare.
- **Authentication is handled by the SDK** — the platform reference holds only the workspace host
  and HTTP path; the Databricks SDK handles authentication (for example, via `databricks auth login`).

The fixture seeds a session-scoped `TEMPORARY VIEW`, so the eval needs only query permissions
and leaves nothing behind in the catalog.

## Write the eval

Create `test_databricks.py`:

```python
--8<-- "examples/04_databricks/test_deterministic.py"
```

The example reads its warehouse connection from the environment, so set these before running:

- `DATABRICKS_SERVER_HOSTNAME`, `DATABRICKS_HTTP_PATH` — your warehouse's host and HTTP path.
  They're just arguments to `databricks_platform()`; pass them as literals if you prefer.
- `DATABRICKS_TOKEN` — read by the Databricks SDK to authenticate. Or use another method it
  supports, e.g. `databricks auth login` for OAuth.

## Run it

```bash
uv run pytest test_databricks.py -q
```

!!! tip "Run it from a clone"
    This is the bundled `examples/04_databricks/` example. If you've cloned the repo, run it
    directly with `uv run pytest examples/04_databricks`.

## Next steps

- [Concepts](../concepts.md) — platforms, scorers, and expected-types in depth.
- [Platforms reference](../reference/platforms.md) — the adapter API.
