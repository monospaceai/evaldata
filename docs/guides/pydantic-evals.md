# Evaluate SQL with Pydantic Evals

Already use [Pydantic Evals](https://ai.pydantic.dev/evals/)? `SqlEquivalence` adds
execution-based SQL scoring to its datasets and reports. It runs generated SQL on your data
platform and checks the result against a gold query or expected rows, so equivalent queries pass
even when their SQL text differs.

## Prerequisites

```bash
uv add "evaldata[pydantic-evals]"
```

## Write the eval

A Pydantic Evals task receives each case's `inputs`; its return value becomes `ctx.output`.
`SqlEquivalence` expects that output to be a non-empty SQL string. The case's `expected_output`
can be a gold SQL string or an evaldata `GoldQuery`, `UntypedResultSet`, or `TypedResultSet`.

Create `test_pydantic_evals.py`:

```python
--8<-- "examples/11_pydantic_evals/test_sql_equivalence.py"
```

The example uses fixed task responses so it runs without a model or network call. The first
response rewrites the gold query with a CTE but returns the same rows, so it passes. The second
uses `COUNT` where the question requires `AVG`, so it fails.

## Run it

```bash
uv run pytest test_pydantic_evals.py -q
```

!!! tip "Run it from a clone"
    This is the bundled `examples/11_pydantic_evals/` example. If you've cloned the repository,
    run it with `uv run --extra pydantic-evals pytest examples/11_pydantic_evals -q`.

## How scoring works

For a gold query, `SqlEquivalence` first checks whether the generated and gold SQL normalize to
the same structure. If that check cannot confirm equivalence, it runs the gold query and compares
the two result sets. An expected result set is compared directly with the generated query's result.

Different results fail with a reason summarizing any row-count, column, type, or value mismatches.
SQL execution errors also produce a failed evaluation rather than escaping from the evaluator.
Invalid case contracts, such as an empty task output or unsupported `expected_output`, raise
`ValueError`.

## Read the report

`evaluate_sync` returns a Pydantic Evals `EvaluationReport`. Each case stores the evaluator's
boolean verdict and explanation under `assertions`:

```python
for case in report.cases:
    result = case.assertions["SqlEquivalence"]
    print(case.name, result.value, result.reason)
```

## Run cases concurrently

Pass `max_concurrency` to let Pydantic Evals score cases in parallel:

```python
report = dataset.evaluate_sync(generate_sql, max_concurrency=8)
```

Each case checks out a session from the platform's connection pool. The pool caps the number of
simultaneous sessions for one platform name; additional cases wait for a session to become
available.

| Engine | Default pool size |
| --- | --- |
| DuckDB | 8 |
| Postgres, Snowflake, BigQuery, Databricks | 4 |
| SQLite | 1 |

Pooled sessions are reused without resetting session-local state. Keep evaluated SQL
side-effect-free: temporary tables, session parameters, roles, and schema changes may be visible
to a later case that receives the same session. Call `close_all()` when the dataset is finished;
the example's fixture does this during teardown.

## Next steps

- [Check semantic equivalence](semantic-equivalence.md): how evaldata decides two queries are equal.
- [Pydantic Evals reference](../reference/pydantic-evals.md): the `SqlEquivalence` API.
