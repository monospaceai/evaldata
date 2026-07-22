"""Tests for `ExecutionAccuracy`, run on SQLite."""

from collections.abc import Iterator

import pytest

from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.scorers import ExecutionAccuracy, QueryRunner, ScoreContext
from evaldata.scorers.execution_accuracy import SCORER_NAME
from evaldata.types import (
    EvalCase,
    GoldQuery,
    ScoreResult,
    SolverSuccess,
    Sql,
    SQLitePlatformRef,
    UntypedResultSet,
)

_OUTPUT = SolverSuccess(output="SELECT ...")


@pytest.fixture
def adapter() -> Iterator[SqliteAdapter]:
    a = SqliteAdapter()
    a.execute("CREATE TABLE items (id INTEGER, name TEXT, price REAL)")
    a.execute("INSERT INTO items VALUES (1, 'apple', 3.0), (2, 'pear', 2.0), (3, 'kiwi', 5.0)")
    yield a
    a.close()


def _case(gold_sql: str) -> EvalCase:
    return EvalCase(
        id="c",
        input="q",
        expected=GoldQuery(sql=gold_sql),
        platform=SQLitePlatformRef(name="sqlite-ea"),
    )


def _score(adapter: SqliteAdapter, gold_sql: str, model_sql: str, **kwargs: object) -> ScoreResult:
    queries = QueryRunner(adapter, Sql(model_sql), "sqlite", None)
    result = queries.run(Sql(model_sql))
    context = ScoreContext(queries=queries)
    return ExecutionAccuracy(**kwargs).score(_case(gold_sql), _OUTPUT, result, context=context)  # ty: ignore[invalid-argument-type]


@pytest.mark.unit
class TestExecutionAccuracy:
    def test_passes_when_rows_match_order_insensitively(self, adapter: SqliteAdapter) -> None:
        # Gold has no ORDER BY, so row order is ignored even though the model reverses it.
        score = _score(adapter, "SELECT id FROM items", "SELECT id FROM items ORDER BY id DESC")
        assert score.verdict == "pass"
        assert score.basis == "observed"
        assert score.scorer == SCORER_NAME

    def test_fails_when_rows_differ_with_diff(self, adapter: SqliteAdapter) -> None:
        score = _score(adapter, "SELECT id FROM items", "SELECT id FROM items WHERE id < 3")
        assert score.verdict == "fail"
        assert score.diff is not None
        assert score.diff.missing_row_count == 1  # id=3 missing from the model result
        assert score.diff.extra_row_count == 0

    def test_order_sensitive_when_gold_has_order_by(self, adapter: SqliteAdapter) -> None:
        gold = "SELECT id FROM items ORDER BY id"
        assert _score(adapter, gold, "SELECT id FROM items ORDER BY id").verdict == "pass"
        assert _score(adapter, gold, "SELECT id FROM items ORDER BY id DESC").verdict == "fail"

    def test_row_order_ignore_overrides_gold_order_by(self, adapter: SqliteAdapter) -> None:
        # Same rows, different order: row_order="ignore" makes it pass despite the gold ORDER BY.
        score = _score(
            adapter,
            "SELECT id FROM items ORDER BY id",
            "SELECT id FROM items ORDER BY id DESC",
            row_order="ignore",
        )
        assert score.verdict == "pass"

    def test_multiplicity_set_compares_distinct_rows(self, adapter: SqliteAdapter) -> None:
        gold = "SELECT name FROM items UNION ALL SELECT name FROM items"  # each name twice
        model = "SELECT name FROM items"  # each name once
        assert _score(adapter, gold, model).verdict == "fail"  # multiset differs
        assert _score(adapter, gold, model, multiplicity="set").verdict == "pass"  # set matches

    def test_by_value_passes_when_columns_reordered(self, adapter: SqliteAdapter) -> None:
        gold = "SELECT id, name FROM items"
        model = "SELECT name, id FROM items"  # same data, columns swapped
        assert _score(adapter, gold, model).verdict == "fail"  # by_position is position-strict
        assert _score(adapter, gold, model, column_alignment="by_value").verdict == "pass"

    def test_by_value_fails_when_data_differs(self, adapter: SqliteAdapter) -> None:
        gold = "SELECT id, name FROM items"
        model = "SELECT name, id FROM items WHERE id < 3"  # column-swapped but missing a row
        assert _score(adapter, gold, model, column_alignment="by_value").verdict == "fail"

    def test_by_value_fails_when_column_counts_differ(self, adapter: SqliteAdapter) -> None:
        gold = "SELECT id, name FROM items"
        model = "SELECT id FROM items"  # one column vs two
        assert _score(adapter, gold, model, column_alignment="by_value").verdict == "fail"

    def test_by_value_passes_with_four_columns(self, adapter: SqliteAdapter) -> None:
        # Four columns exercises the >3 pruning branch; a permutation aligns the swapped pair.
        gold = "SELECT id, name, price, id * 10 AS big FROM items"
        model = "SELECT id, price, name, id * 10 AS big FROM items"  # name/price swapped
        assert _score(adapter, gold, model).verdict == "fail"
        assert _score(adapter, gold, model, column_alignment="by_value").verdict == "pass"

    def test_model_execution_error_fails(self, adapter: SqliteAdapter) -> None:
        score = _score(adapter, "SELECT id FROM items", "SELECT FROM nope")
        assert score.verdict == "fail"
        assert score.explanation is not None
        assert "query execution failed" in score.explanation

    def test_gold_query_failure_is_attributed(self, adapter: SqliteAdapter) -> None:
        score = _score(adapter, "SELECT id FROM does_not_exist", "SELECT id FROM items")
        assert score.verdict == "fail"
        assert score.metadata.get("gold_query_failed") is True

    def test_requires_gold_query_expected(self, adapter: SqliteAdapter) -> None:
        case = EvalCase(
            id="c",
            input="q",
            expected=UntypedResultSet(rows=[{"id": 1}]),
            platform=SQLitePlatformRef(name="sqlite-ea"),
        )
        queries = QueryRunner(adapter, Sql("SELECT id FROM items"), "sqlite", None)
        result = queries.run(Sql("SELECT id FROM items"))
        score = ExecutionAccuracy().score(case, _OUTPUT, result, context=ScoreContext(queries=queries))
        assert score.verdict == "inconclusive"
        assert not score.passed
        assert score.metadata.get("scorer_misconfigured") is True
        assert score.explanation is not None
        assert "GoldQuery" in score.explanation
        assert "UntypedResultSet" in score.explanation

    def test_unparseable_gold_falls_back_to_order_insensitive(self) -> None:
        # When the gold query cannot be parsed, order-sensitivity can't be inferred, so the
        # comparison stays order-insensitive rather than raising.
        assert ExecutionAccuracy()._order_sensitive("this is not valid sql @#$", "sqlite") is False

    def test_diff_samples_are_capped(self, adapter: SqliteAdapter) -> None:
        # Gold yields 10 rows the empty model result all misses; samples cap at 5.
        gold = "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 10) SELECT n FROM t"
        score = _score(adapter, gold, "SELECT id FROM items WHERE 1 = 0")
        assert score.verdict == "fail"
        assert score.diff is not None
        assert score.diff.missing_row_count == 10
        assert len(score.diff.sample_missing_rows) == 5
