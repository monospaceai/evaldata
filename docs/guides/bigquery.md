# Evaluate text-to-SQL on BigQuery

This guide shows how to connect, authenticate, and run SQL evals in BigQuery.

## Prerequisites

```bash
uv add "evaldata[bigquery]"
```

## What this guide covers

- **Data checks.** `row_count`, `not_null`, and `unique` expectations run in BigQuery.
- **Authentication.** `bigquery_platform(...)` stores the project, dataset, and location.
  [Application Default Credentials](#authentication) provide the identity.
- **Test data.** The fixture creates a table for the eval, then drops it.

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

`project` is the Google Cloud project where BigQuery runs and bills query jobs.

`dataset` is the default dataset, so SQL can reference tables without a dataset prefix. `location`
sets the job location, for example `US` or `EU`.

When an eval runs, `resolve(platform)` creates a client with the configured credentials.

## Authentication

`evaldata` resolves credentials through Application Default Credentials (ADC). Set one of these up:

- **gcloud.** Run `gcloud auth application-default login` on your workstation.
- **Service account key.** Set `GOOGLE_APPLICATION_CREDENTIALS` to the path of a service-account
  JSON key file.
- **Workload Identity.** On Google Cloud or in CI with Workload Identity Federation, ADC uses the
  attached identity.

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
`BIGQUERY_LOCATION`. With these set, `evaldata doctor` checks the connection without extra flags.

## Next steps

- [Concepts](../concepts.md): platforms, scorers, and expected types in depth.
- [Platforms reference](../reference/platforms.md): the adapter API.
