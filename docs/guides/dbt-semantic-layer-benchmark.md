# Reproduce dbt's Semantic Layer benchmark

dbt Labs published a benchmark of LLM-generated Semantic Layer queries on an ACME Insurance dataset,
run through dbt Cloud. `evaldata` reproduces it locally on DuckDB — the same dataset, the same
questions, and the same model — as `pytest` tests, scored by resolve-and-compare,
run-and-compare, and an optional grader.

## The result

`gpt-5.3-codex` answers each question at the model's default temperature (many reasoning models do
not accept `temperature=0`), so each question is run 10 times and the pass rate reported.
`gpt-4o-mini` runs once at temperature 0.

| Corpus | Model | Accuracy |
|---|---|---|
| ACME (dbt's suite) | `openai/gpt-5.3-codex` | 96.4% (106/110) |
| ACME (dbt's suite) | `openai/gpt-4o-mini` | 45.5% (5/11) |
| jaffle (authored) | `openai/gpt-5.3-codex` | 100.0% (320/320) |
| jaffle (authored) | `openai/gpt-4o-mini` | 31.2% (10/32) |

Of the 110 ACME runs, 44 were decided by the resolve-and-compare tier and 62 by run-and-compare; the
judge was never needed — so `--no-judge` yields the same number with no grader call.

## How the reproduction is built

- **Dataset and questions.** dbt's ACME project (`dbt-labs/semantic-layer-llm-benchmarking`) is
  ported to dbt-duckdb, and its exact 11-question suite (`dbt-labs/dbt-llm-sl-bench`) is committed as
  `acme_bench.yml`. Both are Apache-2.0 (see the fixture's `NOTICE.md`).
- **Faithful golds.** Each question's gold MetricFlow query returns the same rows as dbt's gold SQL
  on the same warehouse; the e2e asserts this row for row.
- **Sound scoring.** The run-and-compare tier aligns columns by value, compares numbers within a
  tolerance, and accepts a redundant extra grouping column — so a correct answer under a different
  metric label or number format is not marked wrong.

## Run it yourself

From a clone of the repository, with an OpenAI key in the environment:

```bash
# Build the ACME fixture (seeds -> models -> semantic manifest).
uv run --group fixtures bash tests/dbt/fixtures/acme_insurance/regen.sh
uv run --group fixtures dbt build \
  --project-dir tests/dbt/fixtures/acme_insurance \
  --profiles-dir tests/dbt/fixtures/acme_insurance

# Score the suite. Reasoning models need --temperature 1.
uv run --all-extras --group fixtures evaldata sl-bench tests/dbt/fixtures/acme_insurance \
  --model openai/gpt-5.3-codex --temperature 1 \
  --cases tests/dbt/fixtures/acme_insurance/acme_bench.yml \
  --json acme.json
```

`sl-bench` runs the suite once; the table above reports the mean over 10 runs, so a single run
varies by a few points at this temperature.

## Next steps

- [Evaluate dbt Semantic Layer queries](dbt-semantic-layer.md) — the eval workflow on your own project.
- [dbt reference](../reference/dbt.md) — the Semantic Layer types, loaders, and scorers.
