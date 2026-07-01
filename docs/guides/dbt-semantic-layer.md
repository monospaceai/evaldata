# Evaluate dbt Semantic Layer queries

Check AI-generated dbt Semantic Layer (MetricFlow) queries against gold answers — on your own
warehouse, with no dbt Cloud account. `evaldata` reads the metrics and dimensions your project
defines, asks a model to answer each question with a metric query, and scores it with a
cost-ordered cascade of three checks: resolve-and-compare, run-and-compare, and an LLM judge.
The cascade exits as soon as one check reaches a verdict, so cheap checks run first.

## Prerequisites

```bash
uv add "evaldata[dbt,dbt-sl]"
```

The `dbt-sl` extra pulls in `dbt-metricflow`, the toolchain that resolves and runs metric queries.

You also need a built dbt project with a semantic layer (semantic models and metrics). Parse it so
`target/` holds the semantic manifest, and build it so the warehouse has the data to query:

```bash
dbt build   # materialise the models and the time spine
dbt parse   # writes target/semantic_manifest.json
```

## Write the cases file

A metric cases file pairs each question with the gold metric query — the metrics to compute and how
to slice, filter, and limit them:

```yaml
# metric_cases.yml
- question: What is total revenue by month?
  metrics: [revenue]
  group_by: [metric_time__month]

- question: What is the average order value for large orders?
  metrics: [average_order_value]
  where: ["{{ Dimension('order_id__is_large_order') }} = true"]
  id: aov-large-orders
```

Each entry needs a `question` and at least one metric. `group_by` items are dimensions, entities,
or a time dimension with a grain (`metric_time__month`); `where` holds MetricFlow filter
expressions; `order_by`, `limit`, and `id` are optional.

## Run it

```bash
evaldata sl-bench path/to/dbt_project --model openai/gpt-4o-mini --cases metric_cases.yml
```

`evaldata` gives the model the project's metrics and dimensions, asks it for a metric query per
question, and scores each against the gold query. It reports the fraction that match:

```
SL accuracy: 84.0% (21/25)
```

`--model` is any [litellm](https://docs.litellm.ai/docs/providers) model id. Other options:

- `--grader-model ID` — the model for the judge tier; defaults to `--model`.
- `--target-dir DIR` — where the artifacts live, if not `<project>/target`.
- `--profiles-dir DIR` / `--target NAME` — find and select the dbt profile target.
- `--limit N` — run only the first `N` questions.
- `--json PATH` — also write the scores and every result to a JSON file.

## How it's scored

Each question runs through three checks in order; the cascade exits at the first verdict:

1. **Resolve and compare.** MetricFlow resolves both the candidate and gold queries against the
   semantic manifest — filling in default time grains and entity paths the way the warehouse would.
   Queries that resolve to the same form are confirmed equivalent without running anything.
2. **Run and compare.** When the resolved forms differ, both queries run through `mf` and their
   result rows are compared. A verdict is reached from the actual data the warehouse returns.
3. **LLM judge.** An LLM grades the candidate against the gold — useful when the queries can't be
   run (for example, the warehouse is unreachable) or when you want a semantic read of the result.

The cascade exits as soon as one check produces a verdict, so later checks only run when earlier
ones don't decide. The LLM judge is paid for only when it runs.

## Run it in pytest

Run Semantic Layer evals as pytest tests by loading the cases yourself:

```python
import pytest

from evaldata.dbt import (
    load_dbt_metrics,
    metric_layer_equivalence,
    MetricLayerSolver,
    assert_metric_eval,
    platform_from_profile,
)

platform = platform_from_profile("path/to/dbt_project")
cases = load_dbt_metrics("path/to/dbt_project/target", platform=platform, cases="metric_cases.yml")


@pytest.mark.parametrize("case", cases, ids=lambda case: case.id)
def test_sl_question(case):
    solver = MetricLayerSolver("openai/gpt-4o-mini")
    assert_metric_eval(case, solver, scorers=[metric_layer_equivalence("openai/gpt-4o-mini")])
```

`load_dbt_metrics` and `platform_from_profile` return a `DbtError` when the project can't be read,
so check for it before iterating. To compose the cascade yourself, use `MetricSpecEquivalence`,
`MetricResultEquivalence`, and `MetricLayerJudge` with `MetricFirstDecisive`.

## How it works

- MetricFlow itself resolves and runs every query, so a verdict matches what the semantic layer
  would actually return — `evaldata` never reimplements MetricFlow's logic.
- Resolving a query needs only `target/semantic_manifest.json`; running one needs the built
  warehouse the project's dbt profile points at.
- The resolve-and-compare check only confirms equivalence — it never rejects a query on structure
  alone, deferring to running the queries when it can't be sure.

## Next steps

- [Evaluate against a dbt project](dbt.md) — text-to-SQL evals on the same project.
- [Score with an LLM judge](llm-judge.md) — the judge tier in depth.
- [dbt reference](../reference/dbt.md) — the Semantic Layer types, loaders, and scorers.
