# Concepts

An eval is four building blocks: a **case**, a **solver**, one or more **scorers**, and a
**platform**. [`assert_eval`](reference/eval.md) runs the solver, executes its SQL on the
platform, and asserts the scorers pass.

## Cases

A case pairs a question with its expected answer. You declare one with the
[`@eval_case`](reference/eval.md) decorator, which injects a prepared
[`EvalCase`](reference/types.md) as the `pytest` `case` fixture — no `conftest.py` required, since
installing `evaldata` registers its `pytest` plugin.

The `expected` value determines how the result is judged. There are four shapes:

| `expected` shape | Meaning |
| --- | --- |
| `{"rows": [...]}` | **Untyped result set** — compare values only. |
| `{"rows": [...], "schema": [{"name", "type"}]}` | **Typed result set** — values **and** column types. Fails if the right value comes back with the wrong type. |
| `{"kind": "gold_query", "sql": "..."}` | **Gold query** — run a reference query and compare its *executed result*, not its SQL text (execution accuracy). |
| `{"kind": "expectation_suite", "expectations": [...]}` | **Expectation suite** — structural checks (`row_count`, `not_null`, `unique`) instead of exact rows. |

## Solvers

A [solver](reference/solvers.md) is the system under test: it turns a case's question into SQL.

- **`CallableSolver`** wraps a function returning SQL. Use it for fixed SQL (deterministic
  evals) or to plug in any logic you already have.
- **`PromptSolver`** calls a model through [litellm](https://docs.litellm.ai) and expects a
  structured reply containing the SQL. Because litellm normalises providers, the same solver
  drives a local Ollama model and a hosted API model — only the model id changes. Requires the
  `litellm` extra.

Swapping the AI under test is a one-line change:

```python
solver = CallableSolver(lambda c: "SELECT ...")     # fixed SQL
solver = PromptSolver(model="ollama_chat/gemma4")    # local Ollama model
solver = PromptSolver(model="openai/gpt-4o-mini")    # hosted model
```

## Scorers

A [scorer](reference/scorers.md) judges the solver's result against the case's `expected`.

- **`ResultSetEquivalence`** compares result rows — for the untyped, typed, and gold-query
  shapes. See [Equivalence](reference/equivalence.md) for how the row diff works.
- **`SemanticEquivalence`** compares the AI's query against a gold query without running either:
  it confirms a match or returns `unknown` when it can't — it never refutes. `observed_equivalence()`
  composes it with `ResultSetEquivalence`, so a confirmed match skips running the queries while
  everything else is decided by running both and comparing the results. See
  [Check semantic equivalence](guides/semantic-equivalence.md).
- **`ExpectationSuiteScorer`** evaluates an expectation suite's structural checks.
- **`LlmJudge`** asks a grader model to score the AI's SQL against criteria you write, for cases
  where comparing rows or matching syntax isn't enough. It returns a pass/fail verdict with the
  grader's score and rationale. See [Score with an LLM judge](guides/llm-judge.md).

Pass a list to `assert_eval`, so a single case can be scored several ways.

### Verdicts and basis

Every scorer returns a [`ScoreResult`](reference/types.md) whose `verdict` is one of three
values: `pass`, `fail`, or `inconclusive` — the last meaning the scorer couldn't decide either way
(a grader call that errored, a query that couldn't be parsed). Inconclusive doesn't mean the
answer is wrong.

A decided result also carries a `basis` — how strong the evidence behind it is:

- **`proven`** — sound on every dataset, by reasoning about the queries (`SemanticEquivalence`).
- **`observed`** — seen to hold on the data that was run (`ResultSetEquivalence`).
- **`judged`** — a probabilistic judgment from a grader model (`LlmJudge`).

## Platforms

A [platform](reference/platforms.md) is the database the SQL runs on. A platform reference is a
lightweight value (e.g. from `duckdb_platform(...)` or `databricks_platform(...)`); `resolve`
turns it into a live adapter when the eval runs.

Supported adapters today: **DuckDB**, **Postgres**, **Databricks**, and **Snowflake** (a BigQuery
adapter is planned). On a real warehouse the adapter does more than ship SQL — for example, the
Databricks adapter resolves precise column types and pushes expectation and equivalence checks
down into SQL so rows aren't pulled back to compare. Cloud platforms authenticate through their
own driver or SDK, which reads your credentials from the environment — they aren't passed through
the platform reference.
