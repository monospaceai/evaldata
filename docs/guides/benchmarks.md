# Run a text-to-SQL benchmark

Score a model on [Spider](https://yale-lily.github.io/spider) or
[BIRD](https://bird-bench.github.io/) and see its **execution accuracy (EX)** — the fraction of
questions where the model's SQL returns the same result as the gold query.

To be sure the score is right, we compared our scoring against each benchmark's own scoring code
— Spider's `result_eq` and BIRD's set comparison — on every question in the dataset. They agree
everywhere, with a few deliberate exceptions we leave out of that comparison:

- A handful of questions select the same output column name twice. evaldata rejects duplicate
  column names, so it scores those differently.
- A couple of Spider questions return text that isn't valid UTF-8. The official scorer ignores
  the bad bytes; evaldata raises an error instead.

These are design choices, not bugs, and together they're under 0.2% of each dataset.

## Fetch a dataset

`evaldata fetch` downloads a benchmark, checks it against a known checksum, and caches it
locally:

```bash
evaldata fetch spider
evaldata fetch bird
```

The download is pinned to a checksum, so you get the exact bytes we tested against, or the fetch
fails. Re-download with `--force`; choose where it lands with `--cache-dir PATH`.

## Run the benchmark

`evaldata bench` loads the cached dataset, runs a solver that puts the database schema in the
prompt, scores each question, and prints the overall EX:

```bash
evaldata bench spider --model openai/gpt-4o-mini
evaldata bench bird --model openai/gpt-4o-mini --limit 100
```

`--model` is any [litellm](https://docs.litellm.ai/docs/providers) model id. Useful options:

- `--limit N` — run only the first `N` questions (a quick check before a full run).
- `--split dev` — which part of the dataset to load (`dev` by default).
- `--json PATH` — also save a JSON file with the scores and every question's result.
- `path` (positional) — point at an already-unzipped dataset folder instead of the cache.

BIRD tags each question with a difficulty, so the output also breaks the EX down by difficulty.
Example output:

```
EX (bird): 54.8% (841/1534)
EX by difficulty (bird)
difficulty   EX      passed/total
challenge    33.1%   49/148
moderate     48.9%   189/386
simple       60.6%   603/995
```

## How a benchmark is scored

Each benchmark sets `ExecutionAccuracy` up to match its own rules, so the two aren't scored the
same way:

- **Spider** matches columns by value (`column_alignment="by_value"`), so the column order
  doesn't have to line up.
- **BIRD** compares the results as a set (`row_order="ignore"`, `multiplicity="set"`), ignoring
  row order and duplicate rows.

`ExecutionAccuracy` runs both the model's SQL and the gold query and compares the results under
these rules. A question passes when the two match; the EX is the fraction that pass.

## Score your own model

The CLI's solver is a [`PromptSolver`][evaldata.solvers.PromptSolver] that puts each question's
database schema in the prompt. To benchmark something else — your own prompt, a fine-tune, a
multi-step agent — load the cases yourself and pass any [`Solver`][evaldata.solvers.Solver] to
[`run_benchmark`][evaldata.run_benchmark]:

```python
from evaldata import ExecutionAccuracy, load_bird, run_benchmark
from your_system import MySolver

cases = list(load_bird("/path/to/bird", split="dev"))
summary = run_benchmark(
    cases,
    MySolver(),
    scorers=[ExecutionAccuracy(row_order="ignore", multiplicity="set")],  # BIRD's config
    limit=100,
)
print(f"EX: {summary.accuracy:.1%} ({summary.passed}/{summary.total})")
```

[`load_spider`][evaldata.load_spider] and [`load_bird`][evaldata.load_bird] yield `EvalCase`s
with the question as `input` and the gold query as the expected answer, so the cases are ordinary
evals — you can score them with any scorer, not only `ExecutionAccuracy`.

## Try it offline

The bundled `examples/06_benchmark` example builds a tiny Spider-shaped dataset in a temp
directory and runs the same `load_spider` → `run_benchmark` path against a mocked model, so it
needs no download, key, or network:

```python
--8<-- "examples/06_benchmark/test_benchmark.py"
```

```bash
uv run pytest examples/06_benchmark -q
```

## Next steps

- [Concepts](../concepts.md) — solvers, scorers, and expected-types in depth.
- [Scorers reference](../reference/scorers.md) — the `ExecutionAccuracy` API and its options.
