# Examples

Runnable `evaldata` examples using `pytest`: the `@eval_case`
decorator, the injected `case` fixture, and `assert_eval`. Each seeds its own small dataset
via an autouse fixture.

Tiers 01-03 differ in the **solver** — the AI system under test — against the same kind of
cases on a local DuckDB. Swapping tiers is a one-line change:

```python
solver = CallableSolver(lambda c: "SELECT ...")        # 01: fixed SQL
solver = PromptSolver(model="ollama_chat/gemma4")      # 02: local Ollama model
solver = PromptSolver(model="openai/gpt-4o-mini")      # 03: hosted model
```

`04_databricks` instead varies the **platform**: the same kinds of cases run against a real
Databricks SQL Warehouse, with a deterministic solver so the focus is the platform.

`05_llm_judge` instead varies the **scorer**: a deterministic solver supplies the AI SQL so the
only model call is the SQL-equivalence judge deciding what the syntax check cannot.

`06_benchmark` instead varies the **data source**: rather than seeding its own cases, it loads a
text-to-SQL benchmark (Spider-shaped here) and measures execution accuracy with `run_benchmark`.

`07_snowflake`, `08_cortex`, and `09_bigquery` are **live-only**: they run against a real
warehouse or service and so, unlike `03`/`05`/`06`, cannot run without credentials.
`07_snowflake` and `09_bigquery` vary the platform; `08_cortex` varies the solver (Snowflake
Cortex Analyst as the AI under test).

## Tiers

| Dir | Solver | Purpose | Needs |
| --- | --- | --- | --- |
| `01_deterministic` | `CallableSolver` (fixed SQL) | Exercises each expected-type and scorer with no model or network | nothing |
| `02_local_ai` | `PromptSolver` → local Ollama | Runs a self-hosted Ollama model through litellm | `evaldata[litellm]` + `ollama pull gemma4` |
| `03_hosted_ai` | `PromptSolver` → hosted model | Sense-checks the hosted-model plumbing with a mocked reply (no live call) | `evaldata[litellm]` |
| `04_databricks` | `CallableSolver` (fixed SQL) | Runs the same cases against a live Databricks SQL Warehouse | `evaldata[databricks]` + a warehouse |
| `05_llm_judge` | `CallableSolver` (fixed SQL) | Has the SQL-equivalence judge decide what the syntax check can't, with a mocked grader reply (no live call) | `evaldata[litellm]` |
| `06_benchmark` | `PromptSolver` (mocked) | Loads a text-to-SQL benchmark and measures execution accuracy with `run_benchmark` | `evaldata[litellm]` |
| `07_snowflake` | `CallableSolver` (fixed SQL) | Runs the same cases against a live Snowflake warehouse (live-only) | `evaldata[snowflake]` + `SNOWFLAKE_*` credentials |
| `08_cortex` | `CortexAnalystSolver` | Snowflake Cortex Analyst answers each question, scored on your warehouse (live-only) | `evaldata[cortex]` + `SNOWFLAKE_*` credentials |
| `09_bigquery` | `CallableSolver` (fixed SQL) | Runs the same cases against a live BigQuery project (live-only) | `evaldata[bigquery]` + Application Default Credentials |

### 01_deterministic
The solver is a `CallableSolver` returning fixed SQL. `test_golden_questions.py` covers the
expected-types: an untyped result set (values only), a typed result set (values + column
types), a `GoldQuery` (the reference query's *result* is compared, not its SQL text), and an
`ExpectationSuite` (`row_count` / `not_null` / `unique`). `test_semantic_equivalence.py` shows
`SemanticEquivalence` confirming AI SQL that differs syntactically from the gold query but means
the same thing — by comparing normalized syntax trees, without running anything — and
`observed_equivalence()` adding an execution fallback that runs the queries when the trees can't
confirm.

### 02_local_ai
`PromptSolver` calls a self-hosted Ollama model through litellm.
Questions ask for plain column selections, whose output column names come from the table
and are therefore stable, keeping exact-row `ResultSetEquivalence` scoring reliable.
Override the model with `EVALDATA_LOCAL_MODEL` (default `ollama_chat/gemma4`; e.g. a coder
model like `ollama_chat/qwen2.5-coder:1.5b`), and point at a remote Ollama instance with
`OLLAMA_API_BASE`.

### 03_hosted_ai
Mirrors 02 against a hosted model (`openai/gpt-4o-mini` by default, override with
`EVALDATA_HOSTED_MODEL`). The model reply is mocked per question, so it runs
deterministically as a sense-check of the example's plumbing without a live model call or an
API key.

### 04_databricks
The same deterministic cases as 01, executed against a real Databricks SQL Warehouse to show
its platform features:
- **Precise type resolution** — the typed case asserts `amount: DECIMAL(10, 2)`, which holds
  only because evaldata resolves precise column types from the warehouse; the raw driver
  reports a scaleless `DECIMAL`.
- **Warehouse pushdown** — `ExpectationSuite` (`row_count` / `not_null` / `unique`) and
  result-set equivalence run server-side, not by pulling rows back.
- **Secretless platform reference** — the `PlatformRef` holds only `server_hostname` /
  `http_path`; credentials are not part of it.

Set `DATABRICKS_SERVER_HOSTNAME` and `DATABRICKS_HTTP_PATH`, and authenticate with the
Databricks CLI (https://github.com/databricks/cli). It seeds a session-scoped
`TEMPORARY VIEW`, so it needs only query permissions and leaves nothing behind.

### 05_llm_judge
Both cases pick AI SQL the syntax check leaves inconclusive, so the SQL-equivalence judge
decides: one a CTE it confirms, one a wrong filter it refutes. The grader needs a
structured-output-capable hosted model (`openai/gpt-4o-mini` by default, override with
`EVALDATA_HOSTED_MODEL`). Its reply is mocked, so it runs without a live model call or an API key.

### 06_benchmark
Builds a tiny Spider-shaped dataset in a temp directory, loads it with `load_spider`, and runs
`run_benchmark` with an `ExecutionAccuracy` scorer to compute aggregate execution accuracy (EX).
The model reply is mocked per question (two right, one wrong), so EX lands below 100% as a real
run would, without a live call or an API key. To run a real benchmark, `evaldata fetch spider`
(or `bird`) and `evaldata bench spider --model ...` — see the
[benchmark guide](../docs/guides/benchmarks.md).

### 07_snowflake
The same deterministic cases as 01, executed against a live Snowflake warehouse. It seeds a table
in an `EVALDATA_EXAMPLES` database, then shows expectation checks pushed down into SQL, a precise
column type resolved from the warehouse (`AMOUNT` as `DECIMAL(10, 2)`), and execution accuracy
against a gold query. Marked `e2e`, so it is **live-only**: it needs the `snowflake` extra and a
reachable account. Set `SNOWFLAKE_ACCOUNT` (and `SNOWFLAKE_WAREHOUSE` / `SNOWFLAKE_ROLE`) and
configure authentication through the environment — see the
[Snowflake guide](../docs/guides/snowflake.md).

### 08_cortex
`CortexAnalystSolver` sends each question to the Snowflake Cortex Analyst REST endpoint and returns
the SQL it generates; evaldata runs that SQL on Snowflake and scores it against a gold query with
`ExecutionAccuracy`. It seeds a jaffle-shop semantic view to answer against. Marked `e2e`, so it is
**live-only**: it needs the `cortex` extra, a reachable account with the `SNOWFLAKE.CORTEX_USER`
database role, and Snowflake credentials in the environment — see the
[Cortex Analyst guide](../docs/guides/cortex.md). Each run consumes Snowflake credits.

### 09_bigquery

The same deterministic cases as 01, executed against a live BigQuery project. It creates an
`orders_ex09` table in `BIGQUERY_DATASET` (defaulting to `evaldata_examples`), then shows typed
result-set comparison, expectation checks pushed down into SQL, and execution accuracy against a
gold query. Marked `e2e` and `cloud`, so it is **live-only**: install the `bigquery` extra, set
`BIGQUERY_PROJECT`, and configure Application Default Credentials — see the
[BigQuery guide](../docs/guides/bigquery.md).

## Running

```bash
# 01 — no extras:
uv run pytest examples/01_deterministic -p no:randomly -q

# 02 — needs litellm + a pulled model:
uv sync --extra litellm && ollama pull gemma4
uv run pytest examples/02_local_ai -p no:randomly -q

# 03 — runs mocked, no key needed:
uv run pytest examples/03_hosted_ai -q

# 04 — needs the databricks extra + a reachable warehouse (authenticate first via the Databricks CLI):
uv sync --extra databricks
DATABRICKS_SERVER_HOSTNAME=... DATABRICKS_HTTP_PATH=... \
  uv run pytest examples/04_databricks -p no:randomly -q

# 05 — runs mocked, no key needed:
uv run pytest examples/05_llm_judge -q

# 06 — runs mocked, no key needed:
uv run pytest examples/06_benchmark -q

# 07 — live-only: needs the snowflake extra + a reachable account (no mock):
uv sync --extra snowflake
SNOWFLAKE_ACCOUNT=... uv run pytest examples/07_snowflake -m e2e -p no:randomly -q

# 08 — live-only: needs the cortex extra + a reachable account with CORTEX_USER (no mock):
uv sync --extra cortex
SNOWFLAKE_ACCOUNT=... uv run pytest examples/08_cortex -m e2e -p no:randomly -q

# 09 — live-only: needs the bigquery extra + a reachable project (no mock):
uv sync --extra bigquery
BIGQUERY_PROJECT=... uv run pytest examples/09_bigquery -m e2e -p no:randomly -q
```
