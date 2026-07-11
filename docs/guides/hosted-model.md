# Evaluate a hosted model

Run a hosted, API-served model and score its generated SQL. The solver is a `PromptSolver` that
calls the model through [litellm](https://docs.litellm.ai); set the model id and provider
credentials in the environment.

## Prerequisites

```bash
uv add "evaldata[litellm]"
```

## Write the eval

The solver is a `PromptSolver(model=...)`. Create `test_hosted_ai.py`:

```python
--8<-- "examples/03_hosted_ai/test_text_to_sql.py"
```

The example reads the model id from `EVALDATA_HOSTED_MODEL` and passes it to
`PromptSolver(model=...)`. That is the model argument, so you can pass a literal instead.
litellm reads your provider credentials from the environment, e.g. `OPENAI_API_KEY`.

## Run it against the live model

```bash
uv run pytest test_hosted_ai.py -q
```

## Or run it deterministically (no key, no network)

To run this in CI without a live model call, mock the model reply. Add a `conftest.py` next to
your test that returns the structured `{"sql": ...}` the solver expects, matched per question:

```python
--8<-- "examples/03_hosted_ai/conftest.py"
```

With the mock in place it runs offline:

```bash
uv run pytest test_hosted_ai.py -q
```

Remove the `conftest.py` (and set a real `OPENAI_API_KEY`) to evaluate the live model instead.

!!! tip "Run it from a clone"
    This is the bundled `examples/03_hosted_ai/` example. If you've cloned the repo, run it
    directly with `uv run pytest examples/03_hosted_ai` — it runs mocked, with no key needed.

## Next steps

- [Evaluate against Databricks](databricks.md) — run an eval on a real Databricks SQL warehouse.
- [Concepts](../concepts.md) — solvers, scorers, and expected-types in depth.
