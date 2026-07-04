# Getting started

Write and run your first eval against a local DuckDB database — no model and no network, so
there's nothing to set up beyond installing the package.

## Install

```bash
uv add evaldata   # core, includes the DuckDB adapter
```

## The shape of an eval

Every eval is the same four pieces:

- a **case** — a question (`input`) and its `expected` answer, on a `platform`
- a **solver** — the system under test that turns the question into SQL
- one or more **scorers** — how the result is judged against `expected`
- a **platform** — the database the SQL runs on

`evaldata` runs on `pytest`: a case is a test function decorated with `@eval_case`, and
`assert_eval` runs the solver's SQL on the platform and asserts the scorers pass.

## Write your first eval

Create `test_first_eval.py`:

```python
import tempfile
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest

from evaldata import CallableSolver, EvalCase, ResultSetEquivalence, assert_eval, eval_case
from evaldata.platforms import duckdb_platform

_DB = Path(tempfile.mkdtemp()) / "shop.duckdb"
platform = duckdb_platform(name="shop", path=str(_DB))


@pytest.fixture(scope="module", autouse=True)
def _seed() -> Iterator[None]:
    con = duckdb.connect(str(_DB))
    con.execute("CREATE TABLE orders (id INTEGER, customer_id INTEGER, amount DECIMAL(10, 2))")
    con.execute("INSERT INTO orders VALUES (1, 1, 10.00), (2, 1, 5.50), (3, 2, 20.00), (4, 2, 7.25)")
    con.close()
    yield


@eval_case(
    input="What is the total order amount?",
    expected={"rows": [{"total": Decimal("42.75")}]},
    platform=platform,
)
def test_total_order_amount(case: EvalCase) -> None:
    solver = CallableSolver(lambda c: "SELECT sum(amount) AS total FROM orders")
    assert_eval(case, solver, scorers=[ResultSetEquivalence()])
```

Here's what each piece does:

- **`@eval_case(...)`** declares the case and injects a prepared `EvalCase` as the `case`
  fixture. You don't need a `conftest.py` — installing `evaldata` registers its `pytest` plugin.
- **`CallableSolver`** is the simplest solver: a function returning the SQL to run. Here it's
  fixed SQL so the result is deterministic; in a real eval this is where your model goes (see
  the [guides](guides/local-ollama.md)).
- **`ResultSetEquivalence`** scores by comparing the solver's result rows to `expected["rows"]`.
- **`assert_eval`** ties it together: run the solver, execute its SQL on the platform, score,
  and fail the test if a scorer fails.

## Run it

```bash
uv run pytest test_first_eval.py -q
```

A passing run looks like:

```
1 passed
```

It passes because the executed SQL returns `42.75`. Change the expected total and rerun to
watch it fail — that failure is the regression signal you'd catch in CI when a prompt or model
drifts.

## The full set of expected types and scorers

The same pattern covers every expected-type and scorer — an untyped result set, a typed one
(values **and** column types), a *gold query* (compared on its executed result, not its SQL
text), and an `ExpectationSuite` of structural checks:

```python
--8<-- "examples/01_deterministic/test_golden_questions.py"
```

This is the runnable example from `examples/01_deterministic/` in the repo.

## Recap

- An eval is a **case** + a **solver** + **scorers**, run on a **platform**.
- A case is a `@eval_case`-decorated test; `assert_eval` runs the solver's SQL and scores it.
- `CallableSolver` runs fixed SQL — swap in a model with `PromptSolver` to test text-to-SQL.

## Next steps

- Swap the solver for a real model — [a local Ollama model](guides/local-ollama.md) or
  [a hosted model](guides/hosted-model.md).
- Run against a warehouse — [Databricks](guides/databricks.md).
- Score with a grader model — [an LLM judge](guides/llm-judge.md).
- Measure a model on Spider or BIRD — [run a text-to-SQL benchmark](guides/benchmarks.md).
- Understand the building blocks in depth — [Concepts](concepts.md).
