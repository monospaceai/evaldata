# evaldata

[![CI](https://github.com/monospaceai/evaldata/actions/workflows/ci.yml/badge.svg)](https://github.com/monospaceai/evaldata/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen.svg)](https://github.com/monospaceai/evaldata/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Test AI-generated SQL before it reaches production.**

`evaldata` runs evals as ordinary `pytest` tests in your existing CI. It can prove SQL
equivalence without executing queries, fall back to warehouse execution, or use an LLM judge
for ambiguous cases.

## Why evaldata

`evaldata` can often decide SQL equivalence without running the query or calling a grader.
When structure is inconclusive, it falls back to warehouse execution or an LLM judge.

- **Semantic equivalence.** Confirm two queries have the same meaning by comparing their
  structure. No execution, no guessing — when it can't confirm, it returns `unknown`.
- **Execution in your warehouse.** Run the query on DuckDB, Postgres, or Databricks and
  compare the results, accounting for row order, NULLs, float tolerance, and types.
- **It's just `pytest`.** Every eval is a test, run in your suite and your CI on every PR.
  No new runner, notebook, or dashboard.
- **An LLM judge when you need one.** For ambiguous questions, missing reference answers,
  or explanations to grade, use a grader model with explicit criteria.

evaldata reproduces dbt's own Semantic Layer benchmark locally on DuckDB — same dataset, questions,
and model — scoring 96.4% with `gpt-5.3-codex` as `pytest` tests. See
[Reproduce dbt's Semantic Layer benchmark](docs/guides/dbt-semantic-layer-benchmark.md).

## Quickstart

```bash
uv add evaldata   # core, includes the DuckDB adapter
```

An eval is a `pytest` test: a **case** (a question and its expected answer), a **solver**
(the system under test that writes the SQL), and a **scorer** (how the answer is judged).

Below, the AI's SQL is written differently from the reference query — reordered predicates,
different casing — but means the same thing. `observed_equivalence()` proves the match from
the query structure alone; no query runs.

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

## Install

```bash
uv add evaldata                # core (includes the DuckDB adapter)
uv add "evaldata[postgres]"    # + Postgres adapter
uv add "evaldata[databricks]"  # + Databricks adapter
uv add "evaldata[litellm]"     # + litellm, to call a model as the AI under test
```

DuckDB, Postgres, and Databricks are the adapters available today. Snowflake and
BigQuery are planned.

## Documentation

Full documentation: **[monospaceai.github.io/evaldata](https://monospaceai.github.io/evaldata/)**

- [Getting started](https://monospaceai.github.io/evaldata/getting-started/) — write and run your first eval.
- [Concepts](https://monospaceai.github.io/evaldata/concepts/) — cases, solvers, scorers, and platforms.
- Guides — [semantic equivalence](https://monospaceai.github.io/evaldata/guides/semantic-equivalence/), [LLM judge](https://monospaceai.github.io/evaldata/guides/llm-judge/), [local Ollama](https://monospaceai.github.io/evaldata/guides/local-ollama/), [hosted model](https://monospaceai.github.io/evaldata/guides/hosted-model/), [Databricks](https://monospaceai.github.io/evaldata/guides/databricks/).
- [API reference](https://monospaceai.github.io/evaldata/reference/) — the public API, generated from docstrings.

## Examples

Runnable examples in [`examples/`](examples/):

| Example | Shows |
| --- | --- |
| [Showcase](examples/01_deterministic/test_showcase.py) | Semantic equivalence with an execution fallback — no setup |
| [Deterministic](examples/01_deterministic/test_golden_questions.py) | Every expected-type and scorer, with fixed SQL |
| [Local AI](examples/02_local_ai/test_text_to_sql.py) | A self-hosted Ollama model as the AI under test |
| [Hosted AI](examples/03_hosted_ai/test_text_to_sql.py) | A hosted model, mocked so it runs without a key |
| [Databricks](examples/04_databricks/test_deterministic.py) | The same cases on a live Databricks SQL Warehouse |
| [LLM judge](examples/05_llm_judge/test_judged_equivalence.py) | Judged equivalence, mocked so it runs without a key |
| [Benchmark](examples/06_benchmark/test_benchmark.py) | Load a Spider/BIRD dataset and measure execution accuracy |

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
