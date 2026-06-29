# Evaluate against a dbt project

Run text-to-SQL evals against your dbt project. `evaldata` reads the project's compiled artifacts
(`manifest.json` and `catalog.json`) for the schema and warehouse connection, then checks each
answer against a gold query you write — on your own warehouse, with no dbt Cloud account.

## Prerequisites

```bash
uv add "evaldata[dbt]"
```

You also need a built dbt project. Compile it and generate its catalog so `target/` holds both
artifacts:

```bash
dbt build          # or: dbt compile
dbt docs generate  # writes catalog.json with resolved column types
```

`catalog.json` is optional. Without it, `evaldata` uses the column types declared in your model
YAML instead of the types the warehouse resolved.

## Write the cases file

A cases file pairs each question with the SQL whose result is the correct answer:

```yaml
# cases.yml
- question: How many customers placed an order in 2024?
  gold_sql: |
    select count(distinct customer_id) as n
    from customers
    where first_order >= '2024-01-01'
  select: [customers]   # optional: limit the schema shown to the model

- question: What is total revenue by month?
  gold_sql: select date_trunc('month', ordered_at) as month, sum(amount) as revenue from orders group by 1
```

Each entry needs a `question` and a `gold_sql`. `select` limits the schema to named tables, and
`id` names the case; both are optional.

## Run it

```bash
evaldata dbt-bench path/to/dbt_project --model openai/gpt-4o-mini --cases cases.yml
```

`evaldata` reads the warehouse connection from the project's dbt profile, gives the model the
project's schema, runs its SQL against each question, and compares the result to the gold query.
It reports the execution accuracy — the fraction of questions whose result matches:

```
EX (dbt): 72.0% (18/25)
```

`--model` is any [litellm](https://docs.litellm.ai/docs/providers) model id. Other options:

- `--mode model` — skip the cases file; instead take every documented model, asking its
  description as the question and using its compiled SQL as the gold answer.
- `--mode tests` — instead check each documented model's result against its `not_null` and
  `unique` data tests, rather than against a gold query.
- `--target-dir DIR` — where the artifacts live, if not `<project>/target`.
- `--profiles-dir DIR` / `--target NAME` — find and select the dbt profile target.
- `--limit N` — run only the first `N` questions.
- `--json PATH` — also write the scores and every result to a JSON file.

## Check the connection

See whether `evaldata` can reach the project's warehouse:

```bash
evaldata doctor --dbt-project path/to/dbt_project
```

## Run it in pytest

Run dbt evals as pytest tests — with your own prompt, a fine-tune, an agent, or a different
scorer — by loading the cases yourself:

```python
import pytest

from evaldata import ExecutionAccuracy, assert_eval
from evaldata.dbt import load_dbt, platform_from_profile
from evaldata.solvers import SCHEMA_PROMPT_TEMPLATE, PromptSolver

platform = platform_from_profile("path/to/dbt_project")
cases = load_dbt("path/to/dbt_project/target", platform=platform, cases="cases.yml")


@pytest.mark.parametrize("case", cases, ids=lambda case: case.id)
def test_dbt_question(case):
    solver = PromptSolver("openai/gpt-4o-mini", prompt_template=SCHEMA_PROMPT_TEMPLATE)
    assert_eval(case, solver, scorers=[ExecutionAccuracy(row_order="ignore", multiplicity="set")])
```

`load_dbt` and `platform_from_profile` return a `DbtError` when the project can't be read, so
check for it before iterating. The cases are ordinary `EvalCase` objects, so any scorer works.

## How it works

- The warehouse comes from the project's dbt profile. `duckdb` and `postgres` targets are
  supported; a duckdb path is resolved relative to the project.
- The schema given to the model is the project's sources and models as `CREATE TABLE`
  statements, with column types from `catalog.json` and descriptions from your model YAML.
- `ExecutionAccuracy` compares results as a set, ignoring row order and duplicate rows: a
  question passes when the model's SQL and the gold query return the same rows.

## Next steps

- [Concepts](../concepts.md) — solvers, scorers, and expected types in depth.
- [Scorers reference](../reference/scorers.md) — `ExecutionAccuracy` and its options.
- [dbt reference](../reference/dbt.md) — `DbtContext`, `load_dbt`, and `platform_from_profile`.
