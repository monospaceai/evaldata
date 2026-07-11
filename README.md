# evaldata

[![CI](https://github.com/monospaceai/evaldata/actions/workflows/ci.yml/badge.svg)](https://github.com/monospaceai/evaldata/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/monospaceai/evaldata/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Evaluate AI-generated SQL with `pytest`.**

`evaldata` runs text-to-SQL evals in your existing test suite.

It checks semantic equivalence of SQL queries, diffs result sets
in your warehouse, and uses an LLM judge for ambiguous cases.

## Why evaldata

- **Semantic equivalence.** Parse both queries, normalize their ASTs, and
  compare canonical forms. No execution, no LLM — when it can't confirm, it returns
  `unknown`.
- **Execution in your warehouse.** Run the query on DuckDB, Postgres, Databricks, or Snowflake
  and compare the results, accounting for row order, NULLs, float tolerance, and types.
- **It's just `pytest`.** Every eval is a test, run in your suite and your CI on every PR.
  No new runner, notebook, or dashboard.
- **An LLM judge when you need one.** For ambiguous questions, missing reference answers,
  or explanations to grade, use a grader model with explicit criteria.

## Quickstart

```bash
uv add evaldata   # core, includes the DuckDB adapter
```

An eval is a `pytest` test: a **case** (a question and its expected answer), a **solver**
(the system under test that writes the SQL), and a **scorer** (how the answer is judged).

Below, the AI's SQL is written differently from the reference query — reordered predicates,
different casing — but means the same thing. `observed_equivalence()` confirms the match
with AST normalization; no query runs.

```python
from evaldata import CallableSolver, EvalCase, assert_eval, eval_case, observed_equivalence
from evaldata.platforms import duckdb_platform

platform = duckdb_platform(name="shop", path="shop.duckdb")


@eval_case(
    input="Name the US customers with an id above 1.",
    expected={"kind": "gold_query", "sql": "SELECT name FROM customers WHERE country = 'US' AND id > 1"},
    platform=platform,
)
def test_us_customers(case: EvalCase) -> None:
    solver = CallableSolver(lambda c: "select NAME from customers where id > 1 and country = 'US'")
    assert_eval(case, solver, scorers=[observed_equivalence()])
```

```bash
uv run pytest
```

```
 case               result   detail
 ──────────────────────────────────
 test_us_customers  PASS

 1 passed, 0 failed
```

The full runnable version is in
[`examples/01_deterministic/test_showcase.py`](examples/01_deterministic/test_showcase.py).

To test a real model instead of fixed SQL, swap the solver for
`PromptSolver(model="openai/gpt-4o-mini")` (needs the `evaldata[litellm]` extra). To judge
equivalence without a warehouse, swap the scorer for `judged_equivalence(model)`.

## More use cases

- [Evaluate dbt projects](docs/guides/dbt.md) against gold SQL.
- [Evaluate dbt Semantic Layer queries](docs/guides/dbt-semantic-layer.md) against gold MetricFlow queries.
- [Evaluate Snowflake Cortex Analyst](docs/guides/cortex.md) against gold SQL.
- [Reproduce dbt's Semantic Layer benchmark](docs/guides/dbt-semantic-layer-benchmark.md)
  locally on DuckDB.

## Install

```bash
uv add evaldata                # core (includes the DuckDB adapter)
uv add "evaldata[postgres]"    # + Postgres adapter
uv add "evaldata[databricks]"  # + Databricks adapter
uv add "evaldata[snowflake]"   # + Snowflake adapter
uv add "evaldata[cortex]"      # + Snowflake Cortex Analyst solver
uv add "evaldata[litellm]"     # + litellm, to call a model from PromptSolver
```

DuckDB, Postgres, Databricks, and Snowflake are the adapters available today. A BigQuery
adapter is planned.

## Documentation

Full documentation: **[monospaceai.github.io/evaldata](https://monospaceai.github.io/evaldata/)**

- [Getting started](https://monospaceai.github.io/evaldata/getting-started/) — write and run your first eval.
- [Concepts](https://monospaceai.github.io/evaldata/concepts/) — cases, solvers, scorers, and platforms.
- Scoring guides — [semantic equivalence](https://monospaceai.github.io/evaldata/guides/semantic-equivalence/), [LLM judge](https://monospaceai.github.io/evaldata/guides/llm-judge/), [composing scorers](https://monospaceai.github.io/evaldata/guides/composing-scorers/).
- Model guides — [local Ollama](https://monospaceai.github.io/evaldata/guides/local-ollama/), [hosted model](https://monospaceai.github.io/evaldata/guides/hosted-model/).
- Platform guides — [Databricks](https://monospaceai.github.io/evaldata/guides/databricks/), [Snowflake](https://monospaceai.github.io/evaldata/guides/snowflake/), [Cortex Analyst](https://monospaceai.github.io/evaldata/guides/cortex/).
- dbt guides — [dbt project](https://monospaceai.github.io/evaldata/guides/dbt/), [dbt Semantic Layer](https://monospaceai.github.io/evaldata/guides/dbt-semantic-layer/), [reproduce dbt's Semantic Layer benchmark](https://monospaceai.github.io/evaldata/guides/dbt-semantic-layer-benchmark/).
- [Run a text-to-SQL benchmark](https://monospaceai.github.io/evaldata/guides/benchmarks/) — load a Spider/BIRD dataset and measure execution accuracy.
- [API reference](https://monospaceai.github.io/evaldata/reference/) — the public API, generated from docstrings.

## Examples

Runnable examples in [`examples/`](examples/):

| Example | Shows |
| --- | --- |
| [Showcase](examples/01_deterministic/test_showcase.py) | Semantic equivalence with an execution fallback — no setup |
| [Deterministic](examples/01_deterministic/test_golden_questions.py) | Every expected-type and scorer, with fixed SQL |
| [Local AI](examples/02_local_ai/test_text_to_sql.py) | A self-hosted Ollama model |
| [Hosted AI](examples/03_hosted_ai/test_text_to_sql.py) | A hosted model, mocked so it runs without a key |
| [Databricks](examples/04_databricks/test_deterministic.py) | The same cases on a live Databricks SQL Warehouse |
| [LLM judge](examples/05_llm_judge/test_judged_equivalence.py) | Judged equivalence, mocked so it runs without a key |
| [Benchmark](examples/06_benchmark/test_benchmark.py) | Load a Spider/BIRD dataset and measure execution accuracy |
| [Snowflake](examples/07_snowflake/test_deterministic.py) | The same cases on a live Snowflake warehouse — live-only, needs `SNOWFLAKE_*` credentials |
| [Cortex Analyst](examples/08_cortex/test_cortex_analyst.py) | Snowflake Cortex Analyst — live-only, needs `SNOWFLAKE_*` credentials |
| [dbt](examples/10_dbt/test_text_to_sql.py) | A dbt project's text-to-SQL, stubbed so it runs offline |
| [dbt Semantic Layer](examples/10_dbt/test_semantic_layer.py) | dbt Semantic Layer (MetricFlow) queries, scored locally on DuckDB |

See [`examples/README.md`](examples/README.md) for details.

## Contributing

```bash
git clone https://github.com/monospaceai/evaldata.git
cd evaldata
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
