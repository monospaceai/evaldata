# Evaluate a local Ollama model

Run a self-hosted [Ollama](https://ollama.com) model and score its generated SQL against expected
rows on a local DuckDB. The solver is a `PromptSolver` that calls
the model through [litellm](https://docs.litellm.ai).

## Prerequisites

```bash
uv add "evaldata[litellm]"     # PromptSolver + litellm
ollama pull qwen2.5-coder:1.5b # example model
```

## Write the eval

The solver is a `PromptSolver(model=...)` with `temperature=0`. The example uses simple column
selections and `ResultSetEquivalence`.

Create `test_local_ai.py`:

```python
--8<-- "examples/02_local_ai/test_text_to_sql.py"
```

The example reads the model id from `EVALDATA_LOCAL_MODEL` and passes it to
`PromptSolver(model=...)`. You can pass a model id directly instead. If Ollama runs somewhere
other than the default, set `OLLAMA_API_BASE`; litellm reads it.

## Run it

```bash
uv run pytest test_local_ai.py -q
```

A failure means the model produced SQL whose result did not match the expected rows.

!!! tip "Run it from a clone"
    This is the bundled `examples/02_local_ai/` example. If you've cloned the repo, run it
    with `uv run pytest examples/02_local_ai`.

## Next steps

- [Evaluate a hosted model](hosted-model.md): run an API-served model.
- [Concepts](../concepts.md): solvers, scorers, and expected types in depth.
