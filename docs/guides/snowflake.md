# Evaluate against Snowflake

Run evals on Snowflake. The solver returns fixed SQL, and `evaldata` runs the checks as SQL in the
warehouse instead of pulling rows back into Python.

## Prerequisites

```bash
uv add "evaldata[snowflake]"
```

## What it demonstrates

- **Warehouse pushdown** — `row_count`, `not_null`, and `unique` expectations run as SQL in
  Snowflake.
- **Authentication is configured separately** — `snowflake_platform(...)` describes the account,
  warehouse, and role; you set up authentication through environment variables (see
  [Authentication](#authentication)).
- **Temporary setup** — the fixture creates a temporary table for the eval and leaves no table
  behind.

## Configure a connection

Describe your Snowflake connection with `snowflake_platform`:

```python
from evaldata.platforms import snowflake_platform

platform = snowflake_platform(
    name="snowflake",
    account="myorg-myaccount",
    user="EVALDATA_SVC",
    warehouse="COMPUTE_WH",
    role="EVALDATA_ROLE",
    database="EVALDATA",
    schema="PUBLIC",
)
```

`account` is the Snowflake account identifier, for example `myorg-myaccount`. Do not include the
`.snowflakecomputing.com` suffix.

`warehouse`, `role`, `database`, and `schema` set the session defaults. Include `database` and
`schema` when your SQL uses unqualified table names.

When an eval runs, `resolve(platform)` opens the connection using the authentication configured
under [Authentication](#authentication).

## Authentication

- **Password** — set `SNOWFLAKE_PASSWORD`, and pass `user` to `snowflake_platform(...)`.
- **Key pair** — set `SNOWFLAKE_PRIVATE_KEY_FILE` to a PEM-encoded PKCS#8 private key. If the key is
  encrypted, also set `SNOWFLAKE_PRIVATE_KEY_FILE_PWD`.
- **Workload identity (OIDC)** — pass `authenticator="WORKLOAD_IDENTITY"` and
  `workload_identity_provider="OIDC"` to `snowflake_platform(...)`, then set `SNOWFLAKE_TOKEN` to the
  token issued by your CI provider.

## Write the eval

Create `test_snowflake.py`:

```python
from collections.abc import Iterator

import pytest

from evaldata import (
    CallableSolver,
    EvalCase,
    ExpectationSuiteScorer,
    assert_eval,
    eval_case,
)
from evaldata.platforms import resolve, snowflake_platform

pytestmark = [pytest.mark.e2e, pytest.mark.cloud]

_TABLE = "evaldata_orders"
_PLATFORM = snowflake_platform(
    name="snowflake",
    account="myorg-myaccount",
    user="EVALDATA_SVC",
    warehouse="COMPUTE_WH",
    role="EVALDATA_ROLE",
    database="EVALDATA",
    schema="PUBLIC",
)


@pytest.fixture(scope="module", autouse=True)
def _seed_warehouse() -> Iterator[None]:
    adapter = resolve(_PLATFORM)

    for sql in [
        f"CREATE OR REPLACE TEMPORARY TABLE {_TABLE} (id INT, customer STRING)",
        f"INSERT INTO {_TABLE} VALUES (1, 'Ada'), (2, 'Bo'), (3, 'Cy')",
    ]:
        result = adapter.execute(sql)
        if result.error is not None:
            raise RuntimeError(f"failed to seed Snowflake table {_TABLE!r}: {result.error.message}")

    yield


@eval_case(
    input="List every order's id and customer.",
    expected={
        "kind": "expectation_suite",
        "expectations": [
            {"kind": "row_count", "exact": 3},
            {"kind": "not_null", "column": "ID"},
            {"kind": "unique", "column": "ID"},
        ],
    },
    platform=_PLATFORM,
)
def test_expectation_suite_pushdown(case: EvalCase) -> None:
    solver = CallableSolver(lambda c: f"SELECT id, customer FROM {_TABLE}")
    assert_eval(case, solver, scorers=[ExpectationSuiteScorer()])
```

`resolve` reuses one connection per platform `name`, so the eval runs in the same session that
created the temporary table.

Snowflake folds unquoted identifiers to uppercase. The query above returns `ID` and `CUSTOMER`, so
expectations must use `ID`.

## Run it

```bash
uv run pytest test_snowflake.py -q
```

## Check the connection

```bash
evaldata doctor \
  --snowflake-account myorg-myaccount \
  --snowflake-user EVALDATA_SVC \
  --snowflake-warehouse COMPUTE_WH
```

The Snowflake doctor flags also read from environment variables:

- `SNOWFLAKE_ACCOUNT`
- `SNOWFLAKE_USER`
- `SNOWFLAKE_WAREHOUSE`
- `SNOWFLAKE_ROLE`

Once those are set, `evaldata doctor` checks the connection without extra flags.

## Next steps

- [Concepts](../concepts.md) — platforms, scorers, and expected types in depth.
- [Platforms reference](../reference/platforms.md) — the adapter API.
