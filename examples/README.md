# Examples

Runnable, pytest-native `dataeval` examples using the product surface: `@eval_case`
decorator + injected `case` fixture + `assert_eval`. Each file seeds its own
`customers` + `orders` DuckDB in a tempdir via an autouse fixture.

The three tiers differ in the **solver** — the AI system under test — against the same
kind of cases. Swapping tiers is a one-line change:

```python
solver = CallableSolver(lambda c: "SELECT ...")        # 01: fixed SQL
solver = PromptSolver(model="ollama_chat/gemma4")      # 02: local Ollama model
solver = PromptSolver(model="openai/gpt-4o-mini")      # 03: hosted model
```

## Tiers

| Dir | Solver | Purpose | Needs |
| --- | --- | --- | --- |
| `01_deterministic` | `CallableSolver` (fixed SQL) | Exercises each expected-type and scorer with no model or network | nothing |
| `02_local_ai` | `PromptSolver` → local Ollama | Runs a self-hosted Ollama model through litellm | `dataeval[litellm]` + `ollama pull gemma4` |
| `03_hosted_ai` | `PromptSolver` → hosted model | Sense-checks the hosted-model plumbing with a mocked reply (no live call) | `dataeval[litellm]` |

### 01_deterministic
The solver is a `CallableSolver` returning fixed SQL. One file covers the expected-types:
an untyped result set (values only), a typed result set (values + column types), a
`GoldQuery` (the reference query's *result* is compared, not its SQL text), and an
`ExpectationSuite` (`row_count` / `not_null` / `unique`).

### 02_local_ai
`PromptSolver` calls a self-hosted Ollama model through litellm.
Questions ask for plain column selections, whose output column names come from the table
and are therefore stable, keeping exact-row `ResultSetEquivalence` scoring reliable.
Override the model with `DATA_EVAL_LOCAL_MODEL` (default `ollama_chat/gemma4`; e.g. a coder
model like `ollama_chat/qwen2.5-coder:1.5b`), and point at a remote Ollama instance with
`OLLAMA_API_BASE`.

### 03_hosted_ai
Mirrors 02 against a hosted model (`openai/gpt-4o-mini` by default, override with
`DATA_EVAL_HOSTED_MODEL`). The model reply is mocked per question, so it runs
deterministically as a sense-check of the example's plumbing without making a real call or
needing an API key.

## Running

```bash
# 01 — no extras:
uv run pytest examples/01_deterministic -p no:randomly -q

# 02 — needs litellm + a pulled model:
uv sync --extra litellm && ollama pull gemma4
uv run pytest examples/02_local_ai -p no:randomly -q

# 03 — runs mocked, no key needed:
uv run pytest examples/03_hosted_ai -q
```
