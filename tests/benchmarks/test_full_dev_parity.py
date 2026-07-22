"""Full-dev differential harness: `ExecutionAccuracy` vs the official oracle over real dev data.

Runs over the entire dev set for each cached benchmark. For each case the gold SQL is executed,
then deterministic subquery-wrapped transforms stand in as predictions. For every prediction that
executes cleanly, the scorer's verdict is compared to the official comparator (Spider's `result_eq`
and BIRD's `set == set`) over rows fetched by raw `sqlite3` against the same database.

Two cases are held out rather than compared:

- Order-sensitivity divergence: Spider keys off the `'order by'` substring; evaldata parses for a
  top-level `ORDER BY`. The Spider oracle here is driven with evaldata's rule so they agree.
- Duplicate output column names: evaldata's adapter rejects results where two columns share a name
  (`kind="duplicate_columns"`); a positional-tuple oracle cannot model that, so those cases are
  skipped. Likewise a few golds touch non-UTF-8 text; those fail to execute on our side and are
  skipped.

The test fails loudly if a benchmark is not cached. Per-query execution is bounded by a wall-clock
timeout.
"""

import sqlite3
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
import sqlglot
from sqlglot.errors import SqlglotError

from evaldata.loaders.benchmarks.bird import load_bird
from evaldata.loaders.benchmarks.fetch import cached_dataset_path
from evaldata.loaders.benchmarks.spider import load_spider
from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.scorers import ExecutionAccuracy, QueryRunner, ScoreContext
from evaldata.types import EvalCase, ExecutionFailure, GoldQuery, SolverSuccess, Sql, SQLiteConfig, SQLitePlatformRef
from tests._vendor.spider_exec_eval import result_eq

_OUTPUT = SolverSuccess(output=Sql("SELECT ..."))
_QUERY_TIMEOUT_SECONDS = 15.0

_DATASETS: list[tuple[str, Callable[[str], Iterator[EvalCase]]]] = [
    ("bird", load_bird),
    ("spider", load_spider),
]


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
    # Strip a trailing `;` so the gold can be wrapped as a subquery (Spider golds carry one).
    cleaned = gold_sql.strip().removesuffix(";").strip()
    inner = f"({cleaned})"
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
    if isinstance(result, ExecutionFailure) and result.error.kind == "duplicate_columns":
        return None
    case = EvalCase(
        id="c",
        input="q",
        expected=GoldQuery(sql=gold_sql),
        platform=SQLitePlatformRef(name=f"bird-parity:{db_path}", config=SQLiteConfig(path=db_path)),
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


@pytest.mark.e2e
@pytest.mark.timeout(0)
@pytest.mark.parametrize(("dataset", "loader"), _DATASETS, ids=[d[0] for d in _DATASETS])
def test_full_dev_parity(dataset: str, loader: Callable[[str], Iterator[EvalCase]]) -> None:
    """Our scorer agrees with the official oracle on every clean (gold, prediction) over the dev set."""
    root = cached_dataset_path(dataset)
    if root is None:
        # On-demand validation: CI deliberately never downloads these large, flaky-to-fetch
        # datasets, so skip when uncached rather than fail.
        pytest.skip(f"{dataset} not cached; run: evaldata fetch {dataset}")
    cases = list(loader(str(root)))
    comparisons = 0
    skips = 0
    mismatches: list[tuple[str, str, str, str, bool, bool]] = []
    adapters: dict[str, SqliteAdapter] = {}

    try:
        for case in cases:
            expected = case.expected
            assert isinstance(expected, GoldQuery)
            assert isinstance(case.platform, SQLitePlatformRef)
            gold_sql = expected.sql
            db_path = case.platform.config.path
            order_matters = _top_level_order_by(gold_sql)
            adapter = adapters.setdefault(db_path, SqliteAdapter(db_path))

            # Gold with duplicate output column names is unrepresentable in the name-keyed adapter;
            # skip all variants rather than count them as mismatches.
            if isinstance(adapter.execute(gold_sql), ExecutionFailure):
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
        f"\n{dataset} full-dev parity: cases={len(cases)} comparisons={comparisons} "
        f"skips={skips} mismatches={len(mismatches)}"
    )
    assert mismatches == [], f"first mismatches: {mismatches[:5]}"
