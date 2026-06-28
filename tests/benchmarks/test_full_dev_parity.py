"""Full-dev differential harness: `ExecutionAccuracy` vs the official oracle over real BIRD data.

Runs over the entire cached BIRD dev set against the real schemas and SQLite databases. For each
case the gold SQL is executed, then deterministic SQL-safe transforms of the gold (wrapped as a
subquery) stand in as model predictions, exercising pass and edge paths without parsing arbitrary
SQL or calling a model. For every prediction that executes cleanly, the scorer's verdict is
compared to the official comparator (Spider's `result_eq` and BIRD's `set == set`) over rows
fetched by raw `sqlite3` against the same database.

The one documented intentional divergence is held out: Spider keys order-sensitivity off the
`'order by'` substring in the gold SQL, while evaldata parses for a *top-level* `ORDER BY` (a
window `OVER (ORDER BY …)` or a subquery `ORDER BY` does not count). To compare only the
comparison logic, the Spider oracle here is driven with evaldata's own top-level-`ORDER BY` rule,
so the two never disagree merely because they detected order-sensitivity differently.

A second, structural mismatch is also held out: evaldata's adapter keys result rows by output
column name and rejects a result with duplicate output column names (`kind="duplicate_columns"`),
which a positional-tuple oracle cannot model. Some BIRD golds project the same name twice (e.g.
`SELECT T1.name, T2.name …`); the scorer then fails the case because it cannot represent the
gold's rows, while the positional oracle passes. Such cases are skipped (and counted) since the
result is unrepresentable for the scorer rather than a comparison disagreement.

BIRD must already be cached; the test fails loudly if it is not, matching the fail-loud philosophy
of the other e2e tests. Per-query execution is bounded by a wall-clock timeout so a pathological
gold cannot hang the run.
"""

import sqlite3
import threading
from dataclasses import dataclass

import pytest
import sqlglot
from sqlglot.errors import SqlglotError

from evaldata.loaders.benchmarks.bird import load_bird
from evaldata.loaders.benchmarks.fetch import cached_dataset_path
from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.scorers import ExecutionAccuracy, QueryRunner, ScoreContext
from evaldata.types import EvalCase, GoldQuery, PlatformRef, SolverOutput, Sql
from tests._vendor.spider_exec_eval import result_eq

_OUTPUT = SolverOutput(output=Sql("SELECT ..."))
_QUERY_TIMEOUT_SECONDS = 15.0


@dataclass
class _Variant:
    """A model-prediction SQL derived from a gold query, with its id."""

    name: str
    sql: str


def _variants(gold_sql: str, gold_row_count: int) -> list[_Variant]:
    """Build deterministic, SQL-safe prediction variants by wrapping the gold as a subquery.

    Args:
        gold_sql: The gold query text.
        gold_row_count: How many rows the gold returned (gates the row-dropping variant).

    Returns:
        Prediction variants spanning pass and edge paths: identity (pass), distinct (set vs
        multiset), a short limit (missing rows) when the gold returns at least two rows, and an
        explicit ordering.
    """
    inner = f"({gold_sql})"
    variants = [
        _Variant("identity", f"SELECT * FROM {inner} "),
        _Variant("distinct", f"SELECT DISTINCT * FROM {inner} "),
        _Variant("ordered", f"SELECT * FROM {inner} ORDER BY 1"),
    ]
    if gold_row_count >= 2:
        variants.append(_Variant("limited", f"SELECT * FROM {inner} LIMIT {gold_row_count - 1}"))
    return variants


def _execute_bounded(conn: sqlite3.Connection, sql: str) -> list[tuple] | None:
    """Run `sql` and return its rows, or `None` if it errors or exceeds the per-query timeout.

    The query runs on a worker thread; on timeout the connection is interrupted so the worker's
    `execute` unwinds rather than hanging the run.

    Args:
        conn: The read-only SQLite connection.
        sql: The statement to execute.

    Returns:
        The fetched rows as tuples, or `None` on any error or timeout.
    """
    result: list[list[tuple] | None] = [None]

    def _run() -> None:
        try:
            result[0] = conn.execute(sql).fetchall()
        except sqlite3.Error:
            result[0] = None

    worker = threading.Thread(target=_run)
    worker.start()
    worker.join(_QUERY_TIMEOUT_SECONDS)
    if worker.is_alive():
        conn.interrupt()
        worker.join()
        return None
    return result[0]


def _our_verdict(
    adapter: SqliteAdapter, db_path: str, gold_sql: str, pred_sql: str, scorer: ExecutionAccuracy
) -> bool | None:
    """Return whether `scorer` passes `pred_sql` against `gold_sql`, or `None` if unrepresentable.

    `None` signals that the adapter could not represent the prediction's result because it has
    duplicate output column names — a structural limitation that no positional oracle reflects, so
    the comparison is held out rather than counted as a disagreement.
    """
    queries = QueryRunner(adapter, Sql(pred_sql), "sqlite", None)
    result = queries.run(Sql(pred_sql))
    if result.error is not None and result.error.kind == "duplicate_columns":
        return None
    case = EvalCase(
        id="c",
        input="q",
        expected=GoldQuery(sql=gold_sql),
        platform=PlatformRef(name=f"bird-parity:{db_path}", kind="sqlite", config={"path": db_path}),
    )
    context = ScoreContext(queries=queries)
    score = scorer.score(case, _OUTPUT, result, context=context)
    return score.verdict == "pass"


def _top_level_order_by(gold_sql: str) -> bool:
    """Whether `gold_sql`'s top-level statement carries an `ORDER BY`, mirroring the scorer.

    Matches `ExecutionAccuracy._order_sensitive`: a window or subquery `ORDER BY` does not count,
    and an unparseable query is treated as unordered.
    """
    try:
        parsed = sqlglot.parse_one(gold_sql, dialect="sqlite")
    except SqlglotError:
        return False
    return parsed is not None and parsed.args.get("order") is not None


_SPIDER = ExecutionAccuracy(column_alignment="by_value")
_BIRD = ExecutionAccuracy(row_order="ignore", multiplicity="set")


@pytest.fixture(scope="module")
def cases() -> list[EvalCase]:
    """Load the entire cached BIRD dev set, failing loudly if it is not cached."""
    root = cached_dataset_path("bird")
    if root is None:
        pytest.fail("BIRD not cached; run: evaldata fetch bird")
    return list(load_bird(root))


# The full dev set runs thousands of real queries; the global per-test timeout is far too tight,
# and per-query execution is already bounded by `_execute_bounded`.
@pytest.mark.e2e
@pytest.mark.timeout(0)
def test_full_dev_parity(cases: list[EvalCase]) -> None:
    """Our scorer agrees with the official oracle on every clean (gold, prediction) over BIRD dev."""
    comparisons = 0
    skips = 0
    mismatches: list[tuple[str, str, str, str, bool, bool]] = []
    adapters: dict[str, SqliteAdapter] = {}

    try:
        for case in cases:
            expected = case.expected
            assert isinstance(expected, GoldQuery)
            gold_sql = expected.sql
            db_path = str(case.platform.config["path"])
            order_matters = _top_level_order_by(gold_sql)
            adapter = adapters.setdefault(db_path, SqliteAdapter(db_path))

            # The scorer re-runs the gold through the name-keyed adapter; if the gold's projection
            # has duplicate output column names the adapter rejects it, so every variant of this
            # case is unrepresentable for the scorer and held out.
            if adapter.execute(gold_sql).error is not None:
                skips += 1
                continue

            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
            try:
                gold_rows = _execute_bounded(conn, gold_sql)
                if gold_rows is None:
                    skips += 1
                    continue
                for variant in _variants(gold_sql, len(gold_rows)):
                    pred_rows = _execute_bounded(conn, variant.sql)
                    if pred_rows is None:
                        skips += 1
                        continue
                    spider_ours = _our_verdict(adapter, db_path, gold_sql, variant.sql, _SPIDER)
                    bird_ours = _our_verdict(adapter, db_path, gold_sql, variant.sql, _BIRD)
                    if spider_ours is None or bird_ours is None:
                        skips += 2
                        continue
                    # Spider config vs result_eq, and BIRD config vs set==set, share the rows.
                    spider_official = result_eq(pred_rows, gold_rows, order_matters=order_matters)
                    bird_official = set(pred_rows) == set(gold_rows)
                    comparisons += 2
                    if spider_ours != spider_official:
                        mismatches.append(
                            (case.id, f"{variant.name}/spider", gold_sql, variant.sql, spider_ours, spider_official)
                        )
                    if bird_ours != bird_official:
                        mismatches.append(
                            (case.id, f"{variant.name}/bird", gold_sql, variant.sql, bird_ours, bird_official)
                        )
            finally:
                conn.close()
    finally:
        for adapter in adapters.values():
            adapter.close()

    print(
        f"\nBIRD full-dev parity: cases={len(cases)} comparisons={comparisons} "
        f"skips={skips} mismatches={len(mismatches)}"
    )
    assert mismatches == [], f"first mismatches: {mismatches[:5]}"
