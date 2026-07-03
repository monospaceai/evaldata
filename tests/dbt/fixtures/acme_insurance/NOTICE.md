# ACME Insurance fixture — attribution

Copyright dbt Labs, Inc. This fixture is derived from two Apache-2.0 projects, reused under that
license (see `LICENSE.txt`), pinned to the commits below:

- **dbt project + data** — `dbt-labs/semantic-layer-llm-benchmarking`, branch
  `refresh-2025-additional-models` at commit `e4bdee5baeaa9b0ecb8345315c4adfffbeb2f0d1`
  (https://github.com/dbt-labs/semantic-layer-llm-benchmarking) — the seeds, models, and semantic
  layer. The underlying ACME Insurance dataset originates from the data.world "Chat With Your Data"
  benchmark (Sequeda et al.) on the OMG Property & Casualty model.
- **Benchmark questions + gold SQL** — `dbt-labs/dbt-llm-sl-bench` `main` at commit
  `c377468c59ed4b6a5739d87e929cdebc77ba2d32` (https://github.com/dbt-labs/dbt-llm-sl-bench):
  `benchmark_questions.ttl` and the 11-question suite in `src/llm_bench/config/base.py`.

Local adaptations for running on dbt-duckdb (originally targeted dbt Cloud):

- a DuckDB profile (`profiles.yml`);
- `models/omg_semantics/metricflow_time_spine.sql` date literals rewritten from Snowflake
  `to_date('01/01/2000','mm/dd/yyyy')` to `cast('2000-01-01' as date)`, with a
  `metricflow_time_spine.yml` time-spine declaration;
- `Policy_Number` seeded as `bigint` (the identifiers overflow a 32-bit integer).
