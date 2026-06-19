# Examples

Runnable, pytest-native `dataeval` examples using the product surface: the `@eval_case`
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

## Tiers

| Dir | Solver | Purpose | Needs |
| --- | --- | --- | --- |
| `01_deterministic` | `CallableSolver` (fixed SQL) | Exercises each expected-type and scorer with no model or network | nothing |
| `02_local_ai` | `PromptSolver` → local Ollama | Runs a self-hosted Ollama model through litellm | `dataeval[litellm]` + `ollama pull gemma4` |
| `03_hosted_ai` | `PromptSolver` → hosted model | Sense-checks the hosted-model plumbing with a mocked reply (no live call) | `dataeval[litellm]` |
| `04_databricks` | `CallableSolver` (fixed SQL) | Runs the same cases against a live Databricks SQL Warehouse | `dataeval[databricks]` + a warehouse |

### 01_deterministic
The solver is a `CallableSolver` returning fixed SQL. One file covers the expected-types:
an untyped result set (values only), a typed result set (values + column types), a
`GoldQuery` (the reference query's *result* is compared, not its SQL text), and an
`ExpectationSuite` (`row_count` / `not_null` / `unique`).

### 02_local_ai
`PromptSolver` calls a self-hosted Ollama model through litellm.
Questions ask for plain column selections, whose output column names come from the table
and are therefore stable, keeping exact-row `ResultSetEquivalence` scoring reliable.
Override the model with `DATAEVAL_LOCAL_MODEL` (default `ollama_chat/gemma4`; e.g. a coder
model like `ollama_chat/qwen2.5-coder:1.5b`), and point at a remote Ollama instance with
`OLLAMA_API_BASE`.

### 03_hosted_ai
Mirrors 02 against a hosted model (`openai/gpt-4o-mini` by default, override with
`DATAEVAL_HOSTED_MODEL`). The model reply is mocked per question, so it runs
deterministically as a sense-check of the example's plumbing without making a real call or
needing an API key.

### 04_databricks
The same deterministic cases as 01, executed against a real Databricks SQL Warehouse to show
its platform features:
- **Precise type resolution** — the typed case asserts `amount: DECIMAL(10, 2)`, which holds
  only because dataeval resolves precise column types from the warehouse; the raw driver
  reports a scaleless `DECIMAL`.
- **Warehouse pushdown** — `ExpectationSuite` (`row_count` / `not_null` / `unique`) and
  result-set equivalence run server-side, not by pulling rows back.
- **Secretless auth** — the `PlatformRef` holds only `server_hostname` / `http_path`;
  credentials resolve from the ambient environment via the Databricks SDK.

Set `DATABRICKS_SERVER_HOSTNAME` and `DATABRICKS_HTTP_PATH` (plus whatever the Databricks SDK
needs to authenticate, e.g. `DATABRICKS_TOKEN`). It seeds a session-scoped `TEMPORARY VIEW`,
so it needs only query permissions and leaves nothing behind.

## Running

```bash
# 01 — no extras:
uv run pytest examples/01_deterministic -p no:randomly -q

# 02 — needs litellm + a pulled model:
uv sync --extra litellm && ollama pull gemma4
uv run pytest examples/02_local_ai -p no:randomly -q

# 03 — runs mocked, no key needed:
uv run pytest examples/03_hosted_ai -q

# 04 — needs the databricks extra + a reachable warehouse:
uv sync --extra databricks
DATABRICKS_SERVER_HOSTNAME=... DATABRICKS_HTTP_PATH=... DATABRICKS_TOKEN=... \
  uv run pytest examples/04_databricks -p no:randomly -q
```
