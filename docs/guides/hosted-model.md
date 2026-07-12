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
`PromptSolver(model=...)`. You can pass a model id directly instead. litellm reads provider
credentials from the environment, for example `OPENAI_API_KEY`.

## Run it against the live model

```bash
uv run pytest test_hosted_ai.py -q
```

## Use fixed responses in CI

Add a `conftest.py` next to your test that returns the structured `{"sql": ...}` response the
solver expects for each question:

```python
--8<-- "examples/03_hosted_ai/conftest.py"
```

`conftest.py` makes the test use fixed responses:

```bash
uv run pytest test_hosted_ai.py -q
```

Remove `conftest.py` to call the hosted model.

!!! tip "Run it from a clone"
    This is the bundled `examples/03_hosted_ai/` example. If you've cloned the repo, run it
    with `uv run pytest examples/03_hosted_ai`. It includes fixed responses.

## Next steps

- [Evaluate against Databricks](databricks.md): run an eval on a Databricks SQL warehouse.
- [Concepts](../concepts.md): solvers, scorers, and expected types in depth.
