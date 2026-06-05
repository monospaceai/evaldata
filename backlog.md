# Backlog

Larger, not-yet-scheduled design work. Smaller MVP-review points live in `mvp-review.md`.

## North star: test in the warehouse, not in Python

data-eval currently treats the platform as a dumb row-fetcher — it `fetchall()`s the
model's result into Python and runs every check (equivalence, `not_null`, `unique`,
`row_count`) in-process. Established tools (dbt tests/`audit_helper`, Great Expectations
on a SQL backend, SQLMesh `table_diff`) do the opposite: they push every check **into the
warehouse as SQL** and only read back a tiny pass/fail summary.

We want the framework behaviour, for two reasons the team cares about:

1. **The platform is the thing under test.** Users ask "is my data in Snowflake correct?"
   — the answer must be computed by Snowflake's engine, against Snowflake's types, NULL
   rules, collation, and float semantics. A Python re-implementation answers a different,
   weaker question.
2. **Every serialisation boundary is a bug surface.** Driver → Python type coercion (dates,
   decimals, JSON/STRUCT, NULL) can change values before we ever compare them. Testing the
   serialised Python objects tests our serialisation, not the user's data.

So the direction for the items below is: **express each check as SQL that runs against the
model's query in-platform; read back only counts / failing-row samples.** This is a
deliberate reversal of the current in-Python design (including the `not_null` value-scan
shipped in `mvp-review.md` #27, which these tasks supersede).

---

## BL-1 — Foundation: let scorers execute derived SQL in the warehouse

**Status:** `[x]` done (#1) · **Blocks:** BL-2, BL-3

**Why:** Every pushdown below needs to run *derived* queries (wrapping the model's SQL)
against the platform. The current contract can't express that.

**Current state:**
- The model's SQL is executed exactly once in `core/runner.py:assert_eval` via
  `execute_within_budget(...)`, producing a fully-materialised `ExecutionResult` (rows +
  schema).
- The `Scorer` protocol (`scorers/base.py`) is `score(case, output, result)` — it only sees
  the *already-fetched* result, with no handle to the adapter or the SQL, so it cannot run
  a pushdown query.

**Target / approach (resolve in a plan before building):**
- Give scorers a way to execute SQL in-platform — e.g. extend the `Scorer` contract (or add
  a pushdown-scorer variant) to receive the adapter + the model's `Sql`, so a check can run
  `SELECT count(*) FROM (<model sql>) WHERE ...` and read back a scalar/sample.
- Decide the round-trip model: one wrapping query per check (dbt-style; more round-trips,
  scales with the engine) vs. a single combined diagnostic query. Default to one-per-check
  for clarity; measure later.
- Keep errors-as-values: a failed derived query surfaces as a failing `ScoreResult`, like
  today's `result.error` path.
- Preserve the cost-budget / cancellation path (`#25`) for the derived queries too.

**Acceptance criteria:**
- A scorer can run an arbitrary read-only query against the case's platform and get a typed
  result, within the cost budget, with errors-as-values.
- The change is researched against how dbt/GE structure "run check as SQL, read back
  failing rows/count" before locking (per the project's research-design-decisions rule).

---

## BL-2 — Push expectation-suite checks into SQL (`not_null`, `unique`, `row_count`)

**Status:** `[x]` done (#2) · **Depends on:** BL-1 · **Supersedes:** the in-Python checks in
`mvp-review.md` #27

**Why:** These run against the model's live query result — the exact case dbt/GE push to the
warehouse. Doing them in Python is RAM-bound, reimplements native ops, and (for `unique`)
forced the `_hashable_key`/`Counter` hack and a wrong NULL rule.

**Current state (`scorers/expectation_suite.py`):**
- `not_null` → `sum(1 for row in result.rows if row.get(col) is None)` (~line 133).
- `unique` → `Counter(_hashable_key(...) for row in result.rows)` with a repr-tagged
  surrogate key for unhashable cells (`_hashable_key`, ~line 183).
- `row_count` → `len(result.rows)` (~line 77).

**Target (framework-aligned SQL pushdown):**
- `not_null` → `SELECT count(*) FROM (<model sql>) t WHERE <col> IS NULL`; fail if `> 0`.
- `unique` → `SELECT count(*) FROM (SELECT <col> FROM (<model sql>) t GROUP BY <col> HAVING count(*) > 1)`; fail if `> 0`.
- `row_count` → `SELECT count(*) FROM (<model sql>) t`.
- Return failing-row samples by selecting the offending rows, not just a count, so the
  diff message stays useful.
- **Fix the `unique` NULL semantics bug:** the current code flags duplicate NULLs and the
  comment mislabels it as "dbt semantics" — dbt's `unique` does `WHERE col IS NOT NULL`
  (excludes NULLs) and GE ignores them. Align to the framework convention (exclude NULLs)
  and pair with `not_null` for primary-key-style checks. Confirm GE/dbt behaviour and note
  the decision.
- Delete `_hashable_key` / `_render_key` once `unique` is SQL-side (the DB defines equality).

**Schema checks (`column_presence`, `column_type`):** lower priority — they read result
*metadata* (cursor description / catalog), not row data, so they don't cross the row
serialisation boundary. Leave as-is for now, or align with the future catalog-sourced
DDL-nullability work (`mvp-review.md` #27 "deferred"). Note explicitly in the task whichever
is chosen.

**Acceptance criteria:**
- Each of `not_null`/`unique`/`row_count` is computed by SQL the platform runs; Python only
  reads back counts + small failing-row samples.
- `unique` NULL handling matches dbt/GE (documented decision).
- Conformance tests run on DuckDB locally and Postgres via CI/e2e (the checks must hold on
  both engines, since the point is engine-native semantics).

---

## BL-3 — Push result-set equivalence into the warehouse

**Status:** BL-3a `[x]` done (#4) · BL-3b `[x]` done · **Depends on:** BL-1

**Staging:** BL-3a shipped the keyless `EXCEPT ALL` symmetric-diff path (bag semantics,
engine-native NULL equality, typed expected-row materialisation, rounding-based float
tolerance; `null_equality="distinct"` rejected as a failing result). **BL-3b** added the
keyed `FULL OUTER JOIN` path for the cases `EXCEPT ALL` can't express — `null_equality="distinct"`,
an exact `abs(actual-expected) <= float_tolerance` band, and per-column `column_mismatches` —
selected by a non-empty `ComparisonConfig.match_key`. The key join uses plain `=` ANDed per
column (dbt `audit_helper.compare_column_values`; hash-joinable on both engines, unlike a
null-safe operator which Postgres rejects in a `FULL JOIN`), so a `NULL` key never aligns;
non-unique keys are rejected (errors-as-value) and the keyless path remains for bag semantics.

**Why:** Result-set equivalence is the biggest serialisation surface — it fetches every row
into Python and compares with a hand-rolled matcher, so it tests our driver coercion + match
logic rather than the platform's data. Two problems compound: (a) everything is materialised
in Python; (b) the matcher itself is weak.

**Current state:**
- Adapters `fetchall()` the entire result (`platforms/duckdb.py:63`, `postgres.py:74`).
- `equivalence/rows.py:match_multiset` (~lines 30–69) is a greedy **O(n·m)** nested loop —
  for each expected row, linear-scan remaining actual rows for a tolerance-match. The
  docstring already flags it as "best-effort", with a "datacompy-style key-aligned"
  successor planned.
- NULL/float equality is re-implemented in `equivalence/values.py:cells_equal` (Python
  `==`, absolute float tolerance) rather than the engine's `IS NOT DISTINCT FROM` / numeric
  rules.

**Target (framework-aligned):**
- Materialise the authored expected rows *into the platform* (a `VALUES` / CTE, à la dbt
  unit tests) and compute the diff in SQL: `(<model sql>) EXCEPT (<expected>)` +
  `(<expected>) EXCEPT (<model sql>)`, or a keyed `FULL OUTER JOIN` (SQLMesh `table_diff`
  style) when a match key is available. Read back only the mismatch counts + bounded
  samples for `ResultSetDiff`.
- Let the engine define equality (NULL semantics, float/decimal, collation), replacing
  `cells_equal`. Keep `ComparisonConfig` knobs (column order, null equality, float
  tolerance) but translate them into the SQL (e.g. `IS NOT DISTINCT FROM`, rounding /
  `abs(a-b) <= tol`).
- This removes both the fetch-all materialisation for comparison and the O(n·m) matcher.

**Open questions to resolve in the plan:**
- Float tolerance in-SQL across dialects (rounding vs `abs(diff) <= tol`); reuse SQLGlot to
  emit dialect-correct SQL.
- Multiset/bag semantics (duplicate rows) under `EXCEPT` (which is set-based) — may need
  `GROUP BY ... HAVING count` or a row-number key to preserve the current bag behaviour.
- Whether very small expected sets still warrant a fast in-memory path (probably not —
  consistency + engine-native semantics is the whole point).

**Acceptance criteria:**
- Equivalence is computed by SQL the platform runs against an in-warehouse copy of the
  expected rows; Python reads back only counts + bounded samples.
- Engine-native NULL/float/decimal semantics; `ComparisonConfig` knobs honoured via SQL.
- Conformance tests on DuckDB + Postgres (e2e) prove the same case passes/fails identically
  on both engines.

---

## Related (already tracked in `mvp-review.md`)
- **#26** — structured SQL extraction (function calling / structured outputs) instead of
  regex fence-stripping. Not a warehouse-execution item, but the other notable hand-rolled
  spot.
- **#28** — structured per-expectation outcomes (done-ish); will need revisiting once BL-2
  changes how outcomes are produced.
