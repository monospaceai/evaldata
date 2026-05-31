# MVP Review

A running checklist of review points raised file-by-file. We'll go through these
one at a time together. Status: `[ ]` open · `[~]` discussing · `[x]` resolved.

> Note: all original file-by-file points (#1–#21) resolved. Only the deferred
> cross-cutting/design tasks (#22–#26) remain.

## Cross-cutting / design (deferred)

| # | Status | Point |
|---|--------|-------|
| 22 | [ ] | Replace the bare `Schema = list[Column]` alias with a real `Schema` type — likely `RootModel[list[Column]]` (wire-compatible) with Spark-like conveniences (`.names`, `.type_map`, `.columns`, iteration/indexing). Centralises the column-name/type-map projections now done ad-hoc in `compare.py`. **First** research comparable schema APIs (Spark `StructType`, Ibis, pandas, PyArrow, SQLAlchemy) and present a per-question table on the convenience surface before locking. Then `_compare_typed` swaps its comprehensions for `.names`/`.type_map`. |
| 23 | [ ] | Adapters (`duckdb.py:68`, `postgres.py`) silently collapse duplicate output column names via `dict(zip(names, row))` — last-wins data loss with no signal, while `schema_` still lists the duplicate. Research showed the ecosystem convention (GE sets, psycopg `NamedTupleCursor`) is to **reject or surface** duplicates. Detect duplicate names at the adapter boundary and return an `ExecutionResult.error` instead of silently dropping. |
| 24 | [ ] | Reach the new `fail_under = 100` coverage gate (real coverage is ~93.6% once measured correctly). Gaps: (a) Postgres paths in `postgres.py`/`registry.py` need CI-with-Postgres; (b) litellm error branches in `prompt.py:87-92` need exception-forcing tests; (c) defensive/unreachable guards (e.g. `runner.py:45-46`, the `SolverOutput`-validator-guaranteed branch) want `# pragma: no cover`. **Also fix coverage invocation** — must be `coverage run -m pytest` (+ `coverage combine`), not `pytest --cov`: the `pytest11` plugin imports `data_eval` before pytest-cov starts, so import-time lines are missed and it reports a bogus ~57%. |
| 25 | [ ] | No cancellation/timeout enforcement: `CostBudget.max_seconds` exists on `EvalCase` but nothing enforces it. When wiring it, add a `cancel()` method to the `PlatformAdapter` Protocol (dbt-style) so an in-flight query can be aborted when the budget is exceeded. |
| 26 | [ ] | SQL extraction from the model reply uses regex Markdown-fence-stripping (`_FENCE_RE` in `prompt.py`) — standard/pragmatic (Vanna, LangChain do the same) and model-agnostic, but fragile on exotic outputs. Enhancement: use OpenAI **Structured Outputs** / function calling (litellm `response_format`/tools) to return SQL in a typed field where the model supports it, keeping the regex path as fallback for models that don't. |
