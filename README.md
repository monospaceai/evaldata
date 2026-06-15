# dataeval

[![CI](https://github.com/monospaceai/dataeval/actions/workflows/ci.yml/badge.svg)](https://github.com/monospaceai/dataeval/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/monospaceai/dataeval/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**The evaluation framework for AI-generated SQL.**

`pytest`-native evals that execute the SQL your AI writes against real warehouses, so you
catch when a prompt or model change starts producing incorrect answers.

> Status: pre-alpha. The API will change.

## Install (once published)

```bash
uv add dataeval              # core (includes the DuckDB adapter)
uv add "dataeval[postgres]"  # + Postgres adapter
uv add "dataeval[litellm]"   # + litellm, to call a model as the AI under test
```

DuckDB and Postgres are the adapters available today. More warehouse adapters
(Databricks, Snowflake, BigQuery) are planned.

## Examples

Runnable examples in [`examples/`](examples/):

| Example | Shows |
| --- | --- |
| [Deterministic](examples/01_deterministic/test_golden_questions.py) | Every expected-type and scorer with fixed SQL — no model or network |
| [Local AI](examples/02_local_ai/test_text_to_sql.py) | A self-hosted Ollama model as the AI under test, via litellm |
| [Hosted AI](examples/03_hosted_ai/test_text_to_sql.py) | Hosted-model plumbing, mocked so it runs without a live call or key |

See [`examples/README.md`](examples/README.md) for details.

## Develop locally

```bash
git clone https://github.com/monospaceai/dataeval.git
cd dataeval
uv sync                       # core + dev tooling
uv run pre-commit install
just check                    # lint + typecheck + tests with coverage (runs everything)
```

`just check` runs lint, typecheck, and tests with coverage (held at 100%). See the
`justfile` for the full set of commands.

### Platform e2e tests

Adapter conformance for real platforms is marked `e2e`. CI provisions Postgres as a
service container and runs the suite on every push, so the Postgres adapter is exercised
against a real engine on every change.

Run it locally against Postgres with:

```bash
docker compose up -d                  # postgres:17 on localhost:5432
uv run --extra postgres pytest -m e2e # connection via POSTGRES_TEST_* env (defaults match compose)
```
