"""Differential parity: `ExecutionAccuracy` vs the official Spider/BIRD oracles.

Checks that the scorer produces the same pass/fail verdict as the official comparators across
the edge cases that separate them (duplicate rows, reordered columns, row order vs `ORDER BY`,
NULLs, differing column counts, empty results).

- Spider oracle: `result_eq` (Apache-2.0, vendored), with `column_alignment="by_value"`.
- BIRD oracle: `set(pred_rows) == set(gold_rows)`, with `row_order="ignore", multiplicity="set"`.

One intentional divergence is held out: Spider keys order-sensitivity off the `'order by'`
substring while evaldata parses for a top-level `ORDER BY`. Every case below keeps both in
agreement so the only thing under test is the comparison logic.
"""

import sqlite3
from collections.abc import Iterator

import pytest

from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.scorers import ExecutionAccuracy, QueryRunner, ScoreContext
from evaldata.types import EvalCase, GoldQuery, SolverSuccess, Sql, SQLitePlatformRef
from tests._vendor.spider_exec_eval import result_eq

_OUTPUT = SolverSuccess(output=Sql("SELECT ..."))

# (gold_sql, pred_sql, id)
_CASES = [
    ("SELECT a FROM t", "SELECT a FROM t", "identical"),
    ("SELECT a FROM t", "SELECT DISTINCT a FROM t", "duplicate_collapse"),
    ("SELECT a, b FROM t", "SELECT b, a FROM t", "reordered_columns"),
    ("SELECT a FROM t", "SELECT a FROM t ORDER BY a DESC", "reordered_rows_unordered_gold"),
    ("SELECT a FROM t ORDER BY a", "SELECT a FROM t ORDER BY a DESC", "ordered_gold_diff_order"),
    ("SELECT b FROM t", "SELECT b FROM t", "null_passthrough"),
    ("SELECT a FROM t WHERE a < 3", "SELECT a FROM t", "different_rows"),
    ("SELECT a FROM t", "SELECT a, b FROM t", "differing_column_count"),
    ("SELECT a FROM t WHERE a > 99", "SELECT a FROM t WHERE a < 0", "both_empty"),
]


@pytest.fixture
def db_path(tmp_path) -> str:
    """A file-based SQLite DB seeded with duplicates and a NULL, shared by scorer and oracle."""
    path = str(tmp_path / "parity.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [(1, "x"), (1, "x"), (2, "y"), (3, None)])
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def rows(db_path: str):
    """Run `sql` via raw `sqlite3` and return its rows as tuples — what both oracles consume."""

    def _rows(sql: str) -> list[tuple]:
        conn = sqlite3.connect(db_path)
        try:
            return conn.execute(sql).fetchall()
        finally:
            conn.close()

    return _rows


@pytest.fixture
def ours(db_path: str) -> Iterator:
    """Return our scorer's pass/fail as a bool, over the same file-based DB the oracle reads."""
    adapter = SqliteAdapter(db_path)

    def _ours(gold_sql: str, pred_sql: str, **cfg: object) -> bool:
        queries = QueryRunner(adapter, Sql(pred_sql), "sqlite", None)
        result = queries.run(Sql(pred_sql))
        context = ScoreContext(queries=queries)
        case = EvalCase(
            id="c",
            input="q",
            expected=GoldQuery(sql=gold_sql),
            platform=SQLitePlatformRef(name="sqlite-parity"),
        )
        score = ExecutionAccuracy(**cfg).score(case, _OUTPUT, result, context=context)  # ty: ignore[invalid-argument-type]
        return score.verdict == "pass"

    yield _ours
    adapter.close()


@pytest.mark.unit
@pytest.mark.parametrize(("gold", "pred", "case_id"), _CASES, ids=[c[2] for c in _CASES])
class TestOfficialParity:
    def test_spider_parity(self, gold: str, pred: str, case_id: str, rows, ours) -> None:
        official = result_eq(rows(pred), rows(gold), order_matters="order by" in gold.lower())
        mine = ours(gold, pred, column_alignment="by_value")
        assert mine == official

    def test_bird_parity(self, gold: str, pred: str, case_id: str, rows, ours) -> None:
        # BIRD oracle (evaluation.py): set(pred_rows) == set(gold_rows).
        official = set(rows(pred)) == set(rows(gold))
        mine = ours(gold, pred, row_order="ignore", multiplicity="set")
        assert mine == official


@pytest.mark.unit
def test_divergences_are_real(rows, ours) -> None:
    """The matrix genuinely exercises Spider != BIRD on the headline cases."""
    # Duplicate collapse: Spider (bag) fails, BIRD (set) passes.
    gold, pred = "SELECT a FROM t", "SELECT DISTINCT a FROM t"
    spider = result_eq(rows(pred), rows(gold), order_matters=False)
    bird = set(rows(pred)) == set(rows(gold))
    assert spider is False and bird is True
    assert ours(gold, pred, column_alignment="by_value") is False
    assert ours(gold, pred, row_order="ignore", multiplicity="set") is True

    # Reordered columns: Spider (by-value permutation) passes, BIRD (positional set) fails.
    gold, pred = "SELECT a, b FROM t", "SELECT b, a FROM t"
    spider = result_eq(rows(pred), rows(gold), order_matters=False)
    bird = set(rows(pred)) == set(rows(gold))
    assert spider is True and bird is False
    assert ours(gold, pred, column_alignment="by_value") is True
    assert ours(gold, pred, row_order="ignore", multiplicity="set") is False
