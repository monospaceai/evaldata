# evaldata

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

## Install

```bash
uv add evaldata                # core (includes the DuckDB adapter)
uv add "evaldata[postgres]"    # + Postgres adapter
uv add "evaldata[databricks]"  # + Databricks adapter
uv add "evaldata[snowflake]"   # + Snowflake adapter
uv add "evaldata[litellm]"     # + litellm, to call a model as the AI under test
```

DuckDB, Postgres, Databricks, and Snowflake are the adapters available today. A BigQuery adapter
is planned.

## Where to go next

- **[Getting started](getting-started.md)** — write and run your first eval in a few minutes.
- **Guides** — [semantic equivalence](guides/semantic-equivalence.md), [LLM judge](guides/llm-judge.md), [a local Ollama model](guides/local-ollama.md), [a hosted model](guides/hosted-model.md), [Databricks](guides/databricks.md), [Snowflake](guides/snowflake.md).
- **[Concepts](concepts.md)** — the building blocks: cases, solvers, scorers, platforms.
- **[API reference](reference/index.md)** — the public API, generated from docstrings.
