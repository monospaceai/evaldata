# evaldata

**Evaluate AI-generated SQL with `pytest`.**

`evaldata` runs text-to-SQL evals in your existing test suite.

It checks semantic equivalence with SQL AST normalization, diffs result sets
in your warehouse, and uses an LLM judge for ambiguous cases.

## Why evaldata

- **Semantic equivalence.** Parse both queries, normalize their SQL ASTs, and
  compare canonical forms. No execution, no LLM — when it can't confirm, it returns
  `unknown`.
- **Execution in your warehouse.** Run the query on DuckDB, Postgres, Databricks, or Snowflake
  and compare the results, accounting for row order, NULLs, float tolerance, and types.
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
uv add "evaldata[cortex]"      # + Snowflake Cortex Analyst solver
uv add "evaldata[litellm]"     # + litellm, to call a model from PromptSolver
```

DuckDB, Postgres, Databricks, and Snowflake are the adapters available today. A BigQuery adapter
is planned.

## More use cases

- [Evaluate dbt projects](guides/dbt.md) against gold SQL.
- [Evaluate dbt Semantic Layer queries](guides/dbt-semantic-layer.md) against gold MetricFlow queries.
- [Evaluate Snowflake Cortex Analyst](guides/cortex.md) against gold SQL.
- [Reproduce dbt's Semantic Layer benchmark](guides/dbt-semantic-layer-benchmark.md)
  locally on DuckDB.

## Where to go next

- **[Getting started](getting-started.md)** — write and run your first eval in a few minutes.
- **Guides** — [semantic equivalence](guides/semantic-equivalence.md), [LLM judge](guides/llm-judge.md), [a local Ollama model](guides/local-ollama.md), [a hosted model](guides/hosted-model.md), [dbt](guides/dbt.md), [dbt Semantic Layer](guides/dbt-semantic-layer.md), [Databricks](guides/databricks.md), [Snowflake](guides/snowflake.md), [Cortex Analyst](guides/cortex.md).
- **[Concepts](concepts.md)** — the building blocks: cases, solvers, scorers, platforms.
- **[API reference](reference/index.md)** — the public API, generated from docstrings.
