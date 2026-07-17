# Evaluate SQL with Pydantic Evals

[Pydantic Evals](https://ai.pydantic.dev/evals/) is a framework for evaluating the output of AI
systems. `SqlEquivalence` is a drop-in Pydantic Evals `Evaluator` that scores generated SQL the way
evaldata does: it executes the generated query and a reference against a real warehouse and checks
whether they are equivalent. This lets a Pydantic Evals `Dataset` grade text-to-SQL by *result*,
not by string match.

## Prerequisites

```bash
uv add "evaldata[pydantic-evals]"
```

## Write the eval

Each Pydantic Evals `Case` carries the task input and an `expected_output`. For `SqlEquivalence`,
the task returns the generated SQL (as `ctx.output`), and `expected_output` is the reference — a
gold SQL `str`, or an evaldata `GoldQuery`, `UntypedResultSet`, or `TypedResultSet`. Add
`SqlEquivalence` as an evaluator, pointing it at a platform:

```python
from evaldata.platforms.registry import duckdb_platform, resolve
from evaldata.pydantic_evals import SqlEquivalence, close_all
from pydantic_evals import Case, Dataset

# A platform to score against. Seed it with the tables the queries read.
platform = duckdb_platform("jaffle")
resolve(platform).execute(
    "CREATE TABLE orders (id INTEGER, region VARCHAR, amount INTEGER); "
    "INSERT INTO orders VALUES (1, 'east', 50), (2, 'west', 120), (3, 'east', 20)"
)


def task(sql: str) -> str:
    """Stand-in for a text-to-SQL system: here the input already is the SQL to score."""
    return sql


dataset = Dataset(
    name="text-to-sql",
    cases=[
        Case(
            name="totals-by-region",
            inputs="SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
            expected_output="SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
        ),
        Case(
            name="wrong-aggregate",
            inputs="SELECT region, COUNT(amount) AS total FROM orders GROUP BY region",
            expected_output="SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
        ),
    ],
    evaluators=[SqlEquivalence(platform=platform)],
)

report = dataset.evaluate_sync(task)
report.print()
close_all()
```

`SqlEquivalence` judges two queries equivalent when they normalise to the same structure or return
the same rows: it checks structural equivalence first and otherwise diffs the two result sets. A
case whose generated SQL returns different rows fails, with a `reason` naming the row-count
mismatch; invalid SQL fails rather than raising.

## Read the report

`evaluate_sync` returns a Pydantic Evals `EvaluationReport`. Each case exposes the evaluator's
result under `assertions`:

```python
for case in report.cases:
    result = case.assertions["SqlEquivalence"]
    print(case.name, result.value, result.reason)
```

## Concurrency

`SqlEquivalence` works with Pydantic Evals concurrency out of the box: pass `max_concurrency` to
`evaluate_sync` and cases score in parallel. Each case acquires its own warehouse session from a
per-platform-name connection pool for the duration of scoring and returns it afterwards, so nothing
is shared unsafely across threads. Effective parallelism is `min(max_concurrency, pool size)`; extra
cases queue until a session frees up, which also bounds warehouse load.

```python
report = dataset.evaluate_sync(task, max_concurrency=8)
```

Each platform has a default pool size — the most concurrent sessions one platform name opens:

| Engine | Default pool size |
| --- | --- |
| DuckDB | 8 |
| Postgres, Snowflake, BigQuery, Databricks | 4 |
| SQLite | 1 (serial) |

DuckDB pool members are cursors of one shared in-process database, and DuckDB executes their queries
in parallel, so concurrency is a real speed-up (not just overlap). Two caveats when scoring under
concurrency: a case's `cost_budget.max_seconds` is wall-clock and is contended by other in-flight
cases, so a busy pool can make a query overrun sooner; and each dedicated-connection session is
independent, so session-local state (temporary tables, `SET` parameters) does not carry across cases
— seed regular tables (server-side, or the DuckDB shared parent) instead. Call `close_all()` when
done to close every pooled connection.

## Next steps

- [Check semantic equivalence](semantic-equivalence.md): how evaldata decides two queries are equal.
- [Pydantic Evals reference](../reference/pydantic-evals.md): the `SqlEquivalence` API.
