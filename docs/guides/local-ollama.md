# Evaluate a local Ollama model

Run a self-hosted [Ollama](https://ollama.com) model and score its generated SQL against expected
rows on a local DuckDB. The solver is a `PromptSolver` that calls
the model through [litellm](https://docs.litellm.ai).

## Prerequisites

```bash
uv add "evaldata[litellm]"     # PromptSolver + litellm
ollama pull qwen2.5-coder:1.5b # any model you like; a coder model scores best
```

## Write the eval

The solver is a `PromptSolver(model=...)` with `temperature=0` for stable generation. The
questions ask for plain column selections, so the output column names come straight from the
table — which keeps exact-row `ResultSetEquivalence` scoring reliable.

Create `test_local_ai.py`:

```python
--8<-- "examples/02_local_ai/test_text_to_sql.py"
```

The example reads the model id from `EVALDATA_LOCAL_MODEL` and passes it to
`PromptSolver(model=...)`. That is the model argument, so you can pass a literal instead. If
Ollama runs somewhere other than the default, set `OLLAMA_API_BASE` (litellm reads it).

## Run it

```bash
uv run pytest test_local_ai.py -q
```

A failure means the model produced SQL whose result did not match the expected rows.

!!! tip "Run it from a clone"
    This is the bundled `examples/02_local_ai/` example. If you've cloned the repo, run it
    directly with `uv run pytest examples/02_local_ai` — no copying needed.

## Next steps

- [Evaluate a hosted model](hosted-model.md) — run an API-served model.
- [Concepts](../concepts.md) — solvers, scorers, and expected-types in depth.
