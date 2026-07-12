# Evaluate text-to-SQL on BigQuery

Run SQL evals on BigQuery. The example uses fixed SQL so the guide can focus on the adapter:
`evaldata` runs the checks as SQL in BigQuery instead of pulling rows back into Python.

## Prerequisites

```bash
uv add "evaldata[bigquery]"
```

## What this guide covers

- **Query pushdown** — `row_count`, `not_null`, and `unique` expectations run as SQL in BigQuery.
- **Authentication is configured separately** — `bigquery_platform(...)` describes the project,
  dataset, and location; you set up authentication through Application Default Credentials (see
  [Authentication](#authentication)).
- **Temporary setup** — the fixture creates a table for the eval and drops it afterward, leaving
  nothing behind.

## Configure a connection

Describe your BigQuery connection with `bigquery_platform`:

```python
from evaldata.platforms import bigquery_platform

platform = bigquery_platform(
    name="bigquery",
    project="my-project",
    dataset="analytics",
    location="US",
)
```

`project` is the Google Cloud project that jobs run and bill against.

`dataset` sets the default dataset, so SQL can reference tables without a dataset prefix. `location`
pins the location jobs run in, for example `US` or `EU`.

When an eval runs, `resolve(platform)` opens the client using the credentials configured under
[Authentication](#authentication).

## Authentication

`evaldata` resolves credentials through Application Default Credentials (ADC). Set one of these up:

- **gcloud** — run `gcloud auth application-default login` on your workstation.
- **Service account key** — set `GOOGLE_APPLICATION_CREDENTIALS` to the path of a service-account
  JSON key file.
- **Workload Identity** — on Google Cloud or in CI with Workload Identity Federation, ADC resolves
  from the attached identity with no key file.

## Write the eval

Create `test_bigquery.py`:

```python
--8<-- "examples/09_bigquery/test_deterministic.py"
```

`resolve` reuses one client per platform `name`, so the fixture and the evals share a session.

## Run it

```bash
uv run pytest test_bigquery.py -q
```

## Check the connection

```bash
evaldata doctor \
  --bigquery-project my-project \
  --bigquery-dataset analytics \
  --bigquery-location US
```

The BigQuery doctor flags also read from `BIGQUERY_PROJECT`, `BIGQUERY_DATASET`, and
`BIGQUERY_LOCATION`. Once those are set, `evaldata doctor` checks the connection without extra
flags.

## Next steps

- [Concepts](../concepts.md) — platforms, scorers, and expected types in depth.
- [Platforms reference](../reference/platforms.md) — the adapter API.
