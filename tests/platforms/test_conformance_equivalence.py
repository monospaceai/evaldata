"""Conformance battery for result-set equivalence: identical pass/fail + diff counts on DuckDB and Postgres.

Each case builds its model from engine-portable inline `SELECT … UNION ALL`, runs the real
`ResultSetEquivalence` scorer over a real `QueryRunner`, and asserts the same outcome on both
engines — proving the diff uses engine-native `EXCEPT ALL` semantics, not a Python matcher.
"""

from decimal import Decimal
from typing import Literal

import pytest

from dataeval.platforms.base import PlatformAdapter
from dataeval.platforms.duckdb import DuckDBAdapter
from dataeval.scorers import QueryRunner, ResultSetEquivalence, ScoreContext
from dataeval.scorers.sql import Dialect
from dataeval.types import (
    Column,
    ComparisonConfig,
    EvalCase,
    Expected,
    GoldQuery,
    PlatformRef,
    Schema,
    ScoreResult,
    SolverOutput,
    Sql,
    SqlType,
    TypedResultSet,
)

from .conftest import connect_postgres_or_skip


def _duckdb_engine() -> tuple[PlatformAdapter, Dialect]:
    return DuckDBAdapter(), "duckdb"


def _postgres_engine() -> tuple[PlatformAdapter, Dialect]:
    return connect_postgres_or_skip(), "postgres"


@pytest.fixture(
    params=[
        pytest.param(_duckdb_engine, id="duckdb", marks=pytest.mark.unit),
        pytest.param(_postgres_engine, id="postgres", marks=pytest.mark.e2e),
    ],
)
def engine(request: pytest.FixtureRequest) -> tuple[PlatformAdapter, Dialect]:
    """An (adapter, dialect) pair, parametrised across DuckDB (unit) and Postgres (e2e)."""
    return request.param()


def _score(
    engine: tuple[PlatformAdapter, Dialect],
    model: str,
    expected: Expected,
    comparison: ComparisonConfig | None = None,
) -> ScoreResult:
    """Run the model through the adapter, then score its result against `expected`."""
    adapter, dialect = engine
    model_sql = Sql(model)
    kind = "postgres" if dialect == "postgres" else "duckdb"
    case = EvalCase(
        id="c",
        input="q",
        expected=expected,
        platform=PlatformRef(name="x", kind=kind),
        comparison=comparison or ComparisonConfig(),
    )
    result = adapter.execute(model_sql)
    queries = QueryRunner(adapter, model_sql, dialect, None)
    return ResultSetEquivalence().score(
        case, SolverOutput(output=model_sql), result, context=ScoreContext(queries=queries)
    )


def _schema(*pairs: tuple[str, str], dialect: Dialect) -> Schema:
    return Schema(root=[Column(name=n, type=SqlType.parse(t, dialect)) for n, t in pairs])


def _int_rows(values: list[int]) -> str:
    return " UNION ALL ".join(f"SELECT {v} AS n" for v in values)


def test_identical_typed_rows_pass(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": 1}, {"n": 2}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, "SELECT 1 AS n UNION ALL SELECT 2 AS n", expected)
    assert score.passed is True
    assert score.diff is None


def test_one_value_differs_fails(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": 1}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, "SELECT 2 AS n", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1
    assert score.diff.sample_missing_rows == [{"n": 1}]
    assert score.diff.sample_extra_rows == [{"n": 2}]


def test_duplicates_fail_via_bag_semantics(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # actual [{1},{1}] vs expected [{1}]: EXCEPT ALL is a bag, so one extra (a set would pass).
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": 1}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, _int_rows([1, 1]), expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.extra_row_count == 1
    assert score.diff.missing_row_count == 0


def test_unordered_rows_pass(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": 1}, {"n": 2}, {"n": 3}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, _int_rows([2, 1, 3]), expected)
    assert score.passed is True


def test_null_equal_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": None}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, "SELECT CAST(NULL AS INTEGER) AS n", expected)
    assert score.passed is True


def test_null_present_one_side_only_fails(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": None}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, "SELECT 1 AS n", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1


def test_distinct_null_equality_without_key_rejected(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": None}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, "SELECT CAST(NULL AS INTEGER) AS n", expected, ComparisonConfig(null_equality="distinct"))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "requires a match_key" in score.explanation


def test_within_tolerance_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"v": 1.0}], schema=_schema(("v", "DOUBLE"), dialect=dialect))
    score = _score(engine, "SELECT CAST(1.0000000001 AS DOUBLE PRECISION) AS v", expected)
    assert score.passed is True


def test_outside_tolerance_fails(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"v": 1.0}], schema=_schema(("v", "DOUBLE"), dialect=dialect))
    score = _score(engine, "SELECT CAST(1.000001 AS DOUBLE PRECISION) AS v", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1


def test_typed_value_no_string_vs_number_false_mismatch(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"price": Decimal("2.50")}], schema=_schema(("price", "NUMERIC"), dialect=dialect))
    score = _score(engine, "SELECT CAST(2.50 AS NUMERIC) AS price", expected)
    assert score.passed is True


def test_quoted_reserved_column_with_diff_fails(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"order": 1}], schema=_schema(("order", "INTEGER"), dialect=dialect))
    score = _score(engine, 'SELECT 2 AS "order"', expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1


def test_missing_and_unexpected_columns(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(
        rows=[{"a": 1, "b": 2}],
        schema=_schema(("a", "INTEGER"), ("b", "INTEGER"), dialect=dialect),
    )
    score = _score(engine, "SELECT 1 AS a, 3 AS c", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_columns == ["b"]
    assert score.diff.unexpected_columns == ["c"]


def test_derived_query_error_fails_without_raise(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"n": 1}], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(engine, "SELECT n FROM does_not_exist_xyz", expected)
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None


def test_empty_vs_empty_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[], schema=_schema(("n", "INTEGER"), dialect=dialect))
    score = _score(
        engine, "SELECT 1 AS n WHERE 1 = 0" if dialect == "duckdb" else "SELECT 1 AS n WHERE false", expected
    )
    assert score.passed is True


def test_sample_cap_at_twenty(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[], schema=_schema(("n", "INTEGER"), dialect=dialect))
    model = _int_rows(list(range(30)))
    score = _score(engine, model, expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.extra_row_count == 30
    assert len(score.diff.sample_extra_rows) == 20


# ---------- keyed FULL OUTER JOIN path (match_key) ----------


def _id_v_schema(dialect: Dialect, v_type: str = "INTEGER") -> Schema:
    return _schema(("id", "INTEGER"), ("v", v_type), dialect=dialect)


def _keyed(
    key: list[str],
    *,
    null_equality: Literal["equal", "distinct"] = "equal",
    float_tolerance: float = 1e-9,
) -> ComparisonConfig:
    return ComparisonConfig(match_key=key, null_equality=null_equality, float_tolerance=float_tolerance)


def test_keyed_exact_match_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}, {"id": 2, "v": 20}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 2 AS id, 20 AS v UNION ALL SELECT 1, 10", expected, _keyed(["id"]))
    assert score.passed is True
    assert score.diff is None


def test_keyed_per_column_diff_populates_column_mismatches(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}, {"id": 2, "v": 20}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 1 AS id, 10 AS v UNION ALL SELECT 2, 99", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == "v"
    assert score.diff.column_mismatches[0].unexpected_count == 1


def test_keyed_key_only_in_expected_is_missing(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}, {"id": 2, "v": 20}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 1 AS id, 10 AS v", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 0
    assert score.diff.column_mismatches == []
    assert score.diff.sample_missing_rows == [{"id": 2, "v": 20}]


def test_keyed_key_only_in_actual_is_extra(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 1 AS id, 10 AS v UNION ALL SELECT 2, 20", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 1
    assert score.diff.sample_extra_rows == [{"id": 2, "v": 20}]


def test_keyed_equal_null_value_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": None}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 1 AS id, CAST(NULL AS INTEGER) AS v", expected, _keyed(["id"]))
    assert score.passed is True


def test_keyed_distinct_null_value_is_mismatch(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": None}], schema=_id_v_schema(dialect))
    score = _score(
        engine, "SELECT 1 AS id, CAST(NULL AS INTEGER) AS v", expected, _keyed(["id"], null_equality="distinct")
    )
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == "v"


def test_keyed_tolerance_at_band_boundary_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": Decimal("10.0")}], schema=_id_v_schema(dialect, "NUMERIC"))
    score = _score(engine, "SELECT 1 AS id, CAST(10.5 AS NUMERIC) AS v", expected, _keyed(["id"], float_tolerance=0.5))
    assert score.passed is True


def test_keyed_tolerance_just_over_band_fails(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": Decimal("10.0")}], schema=_id_v_schema(dialect, "NUMERIC"))
    score = _score(engine, "SELECT 1 AS id, CAST(10.51 AS NUMERIC) AS v", expected, _keyed(["id"], float_tolerance=0.5))
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == "v"


def test_keyed_composite_key_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    schema = _schema(("a", "INTEGER"), ("b", "VARCHAR"), ("v", "INTEGER"), dialect=dialect)
    expected = TypedResultSet(rows=[{"a": 1, "b": "x", "v": 10}, {"a": 1, "b": "y", "v": 20}], schema=schema)
    model = "SELECT 1 AS a, CAST('x' AS VARCHAR) AS b, 10 AS v UNION ALL SELECT 1, CAST('y' AS VARCHAR), 20"
    score = _score(engine, model, expected, _keyed(["a", "b"]))
    assert score.passed is True


def test_keyed_null_in_key_does_not_align(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # A NULL in a key column is not a valid identifier: it never aligns under `=`, so the
    # expected row is missing and the actual row is extra (dbt audit_helper semantics).
    _, dialect = engine
    schema = _schema(("a", "INTEGER"), ("b", "VARCHAR"), ("v", "INTEGER"), dialect=dialect)
    expected = TypedResultSet(rows=[{"a": 1, "b": None, "v": 10}], schema=schema)
    score = _score(engine, "SELECT 1 AS a, CAST(NULL AS VARCHAR) AS b, 10 AS v", expected, _keyed(["a", "b"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1
    assert score.diff.column_mismatches == []


def test_keyed_duplicate_key_in_actual_rejected(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 1 AS id, 10 AS v UNION ALL SELECT 1, 20", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "not unique in the actual" in score.explanation


def test_keyed_duplicate_key_in_expected_rejected(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}, {"id": 1, "v": 20}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 1 AS id, 10 AS v", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "not unique in the expected" in score.explanation


def test_keyed_absent_key_column_rejected(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT 1 AS id, 10 AS v", expected, _keyed(["nope"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "not present in both" in score.explanation


def test_keyed_quoted_reserved_key_and_column(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    schema = _schema(("order", "INTEGER"), ("v", "INTEGER"), dialect=dialect)
    expected = TypedResultSet(rows=[{"order": 1, "v": 10}], schema=schema)
    score = _score(engine, 'SELECT 1 AS "order", 99 AS v', expected, _keyed(["order"]))
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == "v"


def test_keyed_derived_query_error_fails_without_raise(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": 1, "v": 10}], schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT id, v FROM does_not_exist_xyz", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None


def test_keyed_non_numeric_value_equal_compares_per_column(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    schema = _schema(("id", "INTEGER"), ("name", "VARCHAR"), dialect=dialect)
    expected = TypedResultSet(rows=[{"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}], schema=schema)
    model = "SELECT 1 AS id, CAST('alice' AS VARCHAR) AS name UNION ALL SELECT 2, CAST('wrong' AS VARCHAR)"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == "name"
    assert score.diff.column_mismatches[0].unexpected_count == 1


def test_keyed_non_numeric_distinct_null_value_is_mismatch(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    schema = _schema(("id", "INTEGER"), ("name", "VARCHAR"), dialect=dialect)
    expected = TypedResultSet(rows=[{"id": 1, "name": None}], schema=schema)
    model = "SELECT 1 AS id, CAST(NULL AS VARCHAR) AS name"
    score = _score(engine, model, expected, _keyed(["id"], null_equality="distinct"))
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == "name"


def test_keyed_non_numeric_equal_null_value_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    _, dialect = engine
    schema = _schema(("id", "INTEGER"), ("name", "VARCHAR"), dialect=dialect)
    expected = TypedResultSet(rows=[{"id": 1, "name": None}], schema=schema)
    model = "SELECT 1 AS id, CAST(NULL AS VARCHAR) AS name"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is True


def test_keyed_value_columns_named_like_internal_markers(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # Value columns named `e_present`/`a_present` must not collide with the internal presence
    # markers (ambiguous-column on Postgres / silently wrong counts on DuckDB if they did).
    _, dialect = engine
    schema = _schema(("id", "INTEGER"), ("e_present", "INTEGER"), ("a_present", "INTEGER"), dialect=dialect)
    expected = TypedResultSet(rows=[{"id": 1, "e_present": 10, "a_present": 20}], schema=schema)
    score = _score(engine, "SELECT 1 AS id, 99 AS e_present, 20 AS a_present", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == "e_present"


def test_keyed_multiple_null_keys_not_rejected(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # Two NULL-keyed rows on a side never collide under `=`, so they are not a duplicate key;
    # each surfaces as missing/extra instead of triggering a rejection.
    _, dialect = engine
    expected = TypedResultSet(rows=[{"id": None, "v": 10}, {"id": None, "v": 20}], schema=_id_v_schema(dialect))
    model = "SELECT CAST(NULL AS INTEGER) AS id, 10 AS v UNION ALL SELECT CAST(NULL AS INTEGER), 20"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.explanation is None
    assert score.diff.missing_row_count == 2
    assert score.diff.extra_row_count == 2


# ---------- GoldQuery path (the reference query runs in-warehouse) ----------


def test_gold_matches_model(engine: tuple[PlatformAdapter, Dialect]) -> None:
    expected = GoldQuery(sql="SELECT 1 AS n UNION ALL SELECT 2 AS n")
    score = _score(engine, "SELECT 2 AS n UNION ALL SELECT 1 AS n", expected)
    assert score.passed is True
    assert score.diff is None


def test_gold_differs_from_model(engine: tuple[PlatformAdapter, Dialect]) -> None:
    expected = GoldQuery(sql="SELECT 1 AS n UNION ALL SELECT 2 AS n")
    score = _score(engine, "SELECT 1 AS n UNION ALL SELECT 3 AS n", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1
    assert score.diff.sample_missing_rows == [{"n": 2}]
    assert score.diff.sample_extra_rows == [{"n": 3}]


def test_gold_typed_comparison_detects_type_mismatch(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # Gold yields BIGINT, the model yields INTEGER: the gold schema drives a type mismatch.
    expected = GoldQuery(sql="SELECT CAST(1 AS BIGINT) AS n")
    score = _score(engine, "SELECT CAST(1 AS INTEGER) AS n", expected)
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.type_mismatches) == 1
    assert score.diff.type_mismatches[0].column == "n"


def test_gold_keyed_passes(engine: tuple[PlatformAdapter, Dialect]) -> None:
    expected = GoldQuery(sql="SELECT 1 AS id, 10 AS v UNION ALL SELECT 2, 20")
    score = _score(engine, "SELECT 2 AS id, 20 AS v UNION ALL SELECT 1, 10", expected, _keyed(["id"]))
    assert score.passed is True
    assert score.diff is None


def test_gold_query_error_is_attributed(engine: tuple[PlatformAdapter, Dialect]) -> None:
    expected = GoldQuery(sql="SELECT n FROM does_not_exist_xyz")
    score = _score(engine, "SELECT 1 AS n", expected)
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert score.explanation.startswith("gold query failed:")
    assert score.metadata.get("gold_query_failed") is True
