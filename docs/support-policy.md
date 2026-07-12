# Support & cost policy

evaldata tests its adapters and solvers against real engines. Some of those engines cost money to
run, so this page states which are covered continuously, which are covered on a schedule, and how
hosted-AI solvers are tested without spending on every change.

## How each backend is tested

| Backend | How it is tested | Cadence |
|---|---|---|
| DuckDB, SQLite, Postgres | Live, in a runner container | Every pull request |
| Databricks | Live, against a Free Edition workspace | Every pull request |
| Snowflake | Live, against a Snowflake account over OIDC | Every merge to `main` |
| BigQuery | Live, against a BigQuery project over Workload Identity Federation | Every merge to `main` (or same-repo PR) |
| Cortex Analyst (solver) | Replayed from a recorded response on every change; run live off the pull-request path | Live: manual or scheduled |

The deterministic backend and adapter tests run wherever they are cheap. The shared conformance
suite — one test body run against every adapter — catches dialect differences in core code, so it
runs against Snowflake on every merge to `main`, not only when a Snowflake file changes.

## Hosted AI solvers

A Cortex Analyst message and the warehouse query that runs its SQL both consume Snowflake credits,
so the solver is not called live on every change. Instead:

- The client is tested by **replaying a recorded Cortex Analyst response** — the real request and
  response, captured once and replayed with no network and no credentials. This runs on every pull
  request at no cost.
- The **live accuracy benchmark** runs off the pull-request path, manually or on a schedule.

Recorded responses are scrubbed of credentials and the account host before they are committed, and
a secret scanner blocks any recording that still contains one.

## The accuracy number is a dated snapshot

Cortex Analyst generates SQL with a model, so its output changes over time. Any accuracy number
evaldata publishes is a measurement on a fixed set of questions at a stated date, not a standing
guarantee. The semantic view, seed data, and questions behind the number ship in the repository as
an off-by-default benchmark, so you can reproduce it on your own account.

## Best effort

If access to a paid backend lapses, its live tests move to best effort until access is restored.
The replayed tests and the rest of the suite continue to run unchanged.
