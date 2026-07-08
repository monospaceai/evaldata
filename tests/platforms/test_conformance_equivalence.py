"""Conformance battery for result-set equivalence: identical pass/fail + diff counts across every
registered platform adapter.

Each case builds its model from engine-portable inline `SELECT … UNION ALL`, runs the real
`ResultSetEquivalence` scorer over a real `QueryRunner`, and asserts the same outcome on every
engine — proving the diff uses engine-native bag semantics, not a Python matcher.
"""

from decimal import Decimal
from typing import Any, Literal

import pytest

from evaldata.platforms.base import PlatformAdapter
from evaldata.scorers import QueryRunner, ResultSetEquivalence, ScoreContext
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    Column,
    ComparisonConfig,
    EvalCase,
    Expected,
    GoldQuery,
    PlatformKind,
    PlatformRef,
    ScoreResult,
    SolverOutput,
    Sql,
    SqlType,
    TypedResultSet,
    TypedSchema,
)

from .conftest import conform_name, engine_params, render_model


def _string_type(dialect: Dialect) -> str:
    """The string type name to author so it matches what the engine reports: `STRING` on
    Databricks (whose `VARCHAR` resolves to `string`), `VARCHAR(134217728)` on Snowflake (its
    default max length for an unbounded `VARCHAR` cast), `VARCHAR` elsewhere."""
    if dialect == "databricks":
        return "STRING"
    if dialect == "snowflake":
        return "VARCHAR(134217728)"
    return "VARCHAR"


def _numeric_type(dialect: Dialect) -> str:
    """A scale-bearing fixed-point type: `DECIMAL(10, 2)` on Databricks and Snowflake, whose
    bare `DECIMAL`/`NUMERIC` truncates the scale; `NUMERIC` elsewhere, which reports it bare so
    the authored type must mirror that."""
    return "DECIMAL(10, 2)" if dialect in ("databricks", "snowflake") else "NUMERIC"


def _int_type(dialect: Dialect) -> str:
    """Snowflake types a CAST-to-INTEGER column as NUMBER(38, 0); other engines report INTEGER."""
    return "NUMBER(38, 0)" if dialect == "snowflake" else "INTEGER"


@pytest.fixture(params=engine_params())
def engine(request: pytest.FixtureRequest) -> tuple[PlatformAdapter, PlatformKind]:
    """An (adapter, dialect) pair, parametrised across every registered platform adapter."""
    return request.param()


def _score(
    engine: tuple[PlatformAdapter, PlatformKind],
    model: str,
    expected: Expected,
    comparison: ComparisonConfig | None = None,
    *,
    conform_key: bool = True,
) -> ScoreResult:
    """Run the model through the adapter, then score its result against `expected`.

    `conform_key` folds `comparison.match_key` the way `dialect` folds an unquoted identifier;
    pass `False` when the key column is authored quoted, so it never folds.
    """
    adapter, dialect = engine
    model_sql = Sql(render_model(model, dialect))
    comparison = comparison or ComparisonConfig()
    if comparison.match_key and conform_key:
        comparison = comparison.model_copy(
            update={"match_key": [conform_name(k, dialect) for k in comparison.match_key]}
        )
    case = EvalCase(
        id="c",
        input="q",
        expected=expected,
        platform=PlatformRef(name="x", kind=dialect),
        comparison=comparison,
    )
    result = adapter.execute(model_sql)
    queries = QueryRunner(adapter, model_sql, dialect, None)
    return ResultSetEquivalence().score(
        case, SolverOutput(output=model_sql), result, context=ScoreContext(queries=queries)
    )


def _schema(*pairs: tuple[str, str], dialect: Dialect) -> TypedSchema:
    return TypedSchema(root=[Column(name=conform_name(n, dialect), type=SqlType.parse(t, dialect)) for n, t in pairs])


def _rows(dialect: Dialect, *rows: dict[str, Any]) -> list[dict[str, Any]]:
    """Fold each row dict's keys the way `dialect` folds an unquoted identifier."""
    return [{conform_name(k, dialect): v for k, v in row.items()} for row in rows]


def _int_rows(values: list[int]) -> str:
    return " UNION ALL ".join(f"SELECT CAST({v} AS INTEGER) AS n" for v in values)


def test_identical_typed_rows_pass(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"n": 1}, {"n": 2}), schema=_schema(("n", _int_type(dialect)), dialect=dialect)
    )
    score = _score(engine, "SELECT CAST(1 AS INTEGER) AS n UNION ALL SELECT CAST(2 AS INTEGER) AS n", expected)
    assert score.passed is True
    assert score.diff is None


def test_one_value_differs_fails(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"n": 1}), schema=_schema(("n", _int_type(dialect)), dialect=dialect))
    score = _score(engine, "SELECT CAST(2 AS INTEGER) AS n", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1
    assert score.diff.sample_missing_rows == _rows(dialect, {"n": 1})
    assert score.diff.sample_extra_rows == _rows(dialect, {"n": 2})


def test_duplicates_fail_via_bag_semantics(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # actual [{1},{1}] vs expected [{1}]: the diff is a bag comparison, so one extra (a set would pass).
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"n": 1}), schema=_schema(("n", _int_type(dialect)), dialect=dialect))
    score = _score(engine, _int_rows([1, 1]), expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.extra_row_count == 1
    assert score.diff.missing_row_count == 0


def test_unordered_rows_pass(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"n": 1}, {"n": 2}, {"n": 3}), schema=_schema(("n", _int_type(dialect)), dialect=dialect)
    )
    score = _score(engine, _int_rows([2, 1, 3]), expected)
    assert score.passed is True


def test_null_equal_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"n": None}), schema=_schema(("n", _int_type(dialect)), dialect=dialect)
    )
    score = _score(engine, "SELECT CAST(NULL AS INTEGER) AS n", expected)
    assert score.passed is True


def test_null_present_one_side_only_fails(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"n": None}), schema=_schema(("n", _int_type(dialect)), dialect=dialect)
    )
    score = _score(engine, "SELECT CAST(1 AS INTEGER) AS n", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1


def test_distinct_null_equality_without_key_rejected(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"n": None}), schema=_schema(("n", _int_type(dialect)), dialect=dialect)
    )
    score = _score(engine, "SELECT CAST(NULL AS INTEGER) AS n", expected, ComparisonConfig(null_equality="distinct"))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "requires a match_key" in score.explanation


def test_within_tolerance_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"v": 1.0}), schema=_schema(("v", "DOUBLE"), dialect=dialect))
    score = _score(engine, "SELECT CAST(1.0000000001 AS DOUBLE PRECISION) AS v", expected)
    assert score.passed is True


def test_outside_tolerance_fails(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"v": 1.0}), schema=_schema(("v", "DOUBLE"), dialect=dialect))
    score = _score(engine, "SELECT CAST(1.000001 AS DOUBLE PRECISION) AS v", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1


def test_typed_value_no_string_vs_number_false_mismatch(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    numeric = _numeric_type(dialect)
    expected = TypedResultSet(
        rows=_rows(dialect, {"price": Decimal("2.50")}), schema=_schema(("price", numeric), dialect=dialect)
    )
    score = _score(engine, f"SELECT CAST(2.50 AS {numeric}) AS price", expected)
    assert score.passed is True


def test_quoted_reserved_column_with_diff_fails(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # The model quotes "order" explicitly, so no dialect folds it: the expected side stays
    # literal rather than going through `conform_name` (which only models unquoted folding).
    _, dialect = engine
    schema = TypedSchema(root=[Column(name="order", type=SqlType.parse(_int_type(dialect), dialect))])
    expected = TypedResultSet(rows=[{"order": 1}], schema=schema)
    score = _score(engine, 'SELECT CAST(2 AS INTEGER) AS "order"', expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1


def test_missing_and_unexpected_columns(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"a": 1, "b": 2}),
        schema=_schema(("a", _int_type(dialect)), ("b", _int_type(dialect)), dialect=dialect),
    )
    score = _score(engine, "SELECT CAST(1 AS INTEGER) AS a, CAST(3 AS INTEGER) AS c", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_columns == [conform_name("b", dialect)]
    assert score.diff.unexpected_columns == [conform_name("c", dialect)]


def test_derived_query_error_fails_without_raise(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"n": 1}), schema=_schema(("n", _int_type(dialect)), dialect=dialect))
    score = _score(engine, "SELECT n FROM does_not_exist_xyz", expected)
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None


def test_empty_vs_empty_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[], schema=_schema(("n", _int_type(dialect)), dialect=dialect))
    model = (
        "SELECT CAST(1 AS INTEGER) AS n WHERE 1 = 0"
        if dialect == "duckdb"
        else "SELECT CAST(1 AS INTEGER) AS n WHERE false"
    )
    score = _score(engine, model, expected)
    assert score.passed is True


def test_sample_cap_at_twenty(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=[], schema=_schema(("n", _int_type(dialect)), dialect=dialect))
    model = _int_rows(list(range(30)))
    score = _score(engine, model, expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.extra_row_count == 30
    assert len(score.diff.sample_extra_rows) == 20


def test_keyless_columns_named_like_internal_markers(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # A result whose columns are named `cnt`/`rn` must not collide with the bag diff's internal
    # count/row-number columns (`COUNT(*) AS cnt` is a common alias). Expected has the row twice,
    # actual once, so the bag comparison reports one missing.
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"cnt": 5, "rn": 7}, {"cnt": 5, "rn": 7}),
        schema=_schema(("cnt", _int_type(dialect)), ("rn", _int_type(dialect)), dialect=dialect),
    )
    score = _score(engine, "SELECT CAST(5 AS INTEGER) AS cnt, CAST(7 AS INTEGER) AS rn", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 0
    assert score.diff.sample_missing_rows == _rows(dialect, {"cnt": 5, "rn": 7})


# ---------- keyed FULL OUTER JOIN path (match_key) ----------


def _id_v_schema(dialect: Dialect, v_type: str | None = None) -> TypedSchema:
    return _schema(("id", _int_type(dialect)), ("v", v_type or _int_type(dialect)), dialect=dialect)


def _keyed(
    key: list[str],
    *,
    null_equality: Literal["equal", "distinct"] = "equal",
    float_tolerance: float = 1e-9,
) -> ComparisonConfig:
    return ComparisonConfig(match_key=key, null_equality=null_equality, float_tolerance=float_tolerance)


def test_keyed_exact_match_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}, {"id": 2, "v": 20}), schema=_id_v_schema(dialect))
    model = (
        "SELECT CAST(2 AS INTEGER) AS id, CAST(20 AS INTEGER) AS v "
        "UNION ALL SELECT CAST(1 AS INTEGER), CAST(10 AS INTEGER)"
    )
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is True
    assert score.diff is None


def test_keyed_per_column_diff_populates_column_mismatches(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}, {"id": 2, "v": 20}), schema=_id_v_schema(dialect))
    model = (
        "SELECT CAST(1 AS INTEGER) AS id, CAST(10 AS INTEGER) AS v "
        "UNION ALL SELECT CAST(2 AS INTEGER), CAST(99 AS INTEGER)"
    )
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == conform_name("v", dialect)
    assert score.diff.column_mismatches[0].unexpected_count == 1


def test_keyed_key_only_in_expected_is_missing(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}, {"id": 2, "v": 20}), schema=_id_v_schema(dialect))
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(10 AS INTEGER) AS v"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 0
    assert score.diff.column_mismatches == []
    assert score.diff.sample_missing_rows == _rows(dialect, {"id": 2, "v": 20})


def test_keyed_key_only_in_actual_is_extra(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}), schema=_id_v_schema(dialect))
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(10 AS INTEGER) AS v UNION ALL SELECT CAST(2 AS INTEGER), CAST(20 AS INTEGER)"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 1
    assert score.diff.sample_extra_rows == _rows(dialect, {"id": 2, "v": 20})


def test_keyed_equal_null_value_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": None}), schema=_id_v_schema(dialect))
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(NULL AS INTEGER) AS v"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is True


def test_keyed_distinct_null_value_is_mismatch(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": None}), schema=_id_v_schema(dialect))
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(NULL AS INTEGER) AS v"
    score = _score(engine, model, expected, _keyed(["id"], null_equality="distinct"))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == conform_name("v", dialect)


def test_keyed_tolerance_at_band_boundary_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    numeric = _numeric_type(dialect)
    expected = TypedResultSet(
        rows=_rows(dialect, {"id": 1, "v": Decimal("10.0")}), schema=_id_v_schema(dialect, numeric)
    )
    model = f"SELECT CAST(1 AS INTEGER) AS id, CAST(10.5 AS {numeric}) AS v"
    score = _score(engine, model, expected, _keyed(["id"], float_tolerance=0.5))
    assert score.passed is True


def test_keyed_tolerance_just_over_band_fails(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    numeric = _numeric_type(dialect)
    expected = TypedResultSet(
        rows=_rows(dialect, {"id": 1, "v": Decimal("10.0")}), schema=_id_v_schema(dialect, numeric)
    )
    model = f"SELECT CAST(1 AS INTEGER) AS id, CAST(10.51 AS {numeric}) AS v"
    score = _score(engine, model, expected, _keyed(["id"], float_tolerance=0.5))
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == conform_name("v", dialect)


def test_keyed_composite_key_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    schema = _schema(
        ("a", _int_type(dialect)), ("b", _string_type(dialect)), ("v", _int_type(dialect)), dialect=dialect
    )
    expected = TypedResultSet(
        rows=_rows(dialect, {"a": 1, "b": "x", "v": 10}, {"a": 1, "b": "y", "v": 20}), schema=schema
    )
    model = (
        "SELECT CAST(1 AS INTEGER) AS a, CAST('x' AS VARCHAR) AS b, CAST(10 AS INTEGER) AS v "
        "UNION ALL SELECT CAST(1 AS INTEGER), CAST('y' AS VARCHAR), CAST(20 AS INTEGER)"
    )
    score = _score(engine, model, expected, _keyed(["a", "b"]))
    assert score.passed is True


def test_keyed_null_in_key_does_not_align(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # A NULL in a key column is not a valid identifier: it never aligns under `=`, so the
    # expected row is missing and the actual row is extra (dbt audit_helper semantics).
    _, dialect = engine
    schema = _schema(
        ("a", _int_type(dialect)), ("b", _string_type(dialect)), ("v", _int_type(dialect)), dialect=dialect
    )
    expected = TypedResultSet(rows=_rows(dialect, {"a": 1, "b": None, "v": 10}), schema=schema)
    model = "SELECT CAST(1 AS INTEGER) AS a, CAST(NULL AS VARCHAR) AS b, CAST(10 AS INTEGER) AS v"
    score = _score(engine, model, expected, _keyed(["a", "b"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1
    assert score.diff.column_mismatches == []


def test_keyed_duplicate_key_in_actual_rejected(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}), schema=_id_v_schema(dialect))
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(10 AS INTEGER) AS v UNION ALL SELECT CAST(1 AS INTEGER), CAST(20 AS INTEGER)"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "not unique in the actual" in score.explanation


def test_keyed_duplicate_key_in_expected_rejected(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}, {"id": 1, "v": 20}), schema=_id_v_schema(dialect))
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(10 AS INTEGER) AS v"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "not unique in the expected" in score.explanation


def test_keyed_absent_key_column_rejected(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}), schema=_id_v_schema(dialect))
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(10 AS INTEGER) AS v"
    score = _score(engine, model, expected, _keyed(["nope"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert "not present in both" in score.explanation


def test_keyed_quoted_reserved_key_and_column(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # The model quotes "order" explicitly, so the key column never folds; only the unquoted
    # `v` column does. Built without `_schema`/`_rows` to avoid folding the quoted key.
    _, dialect = engine
    v = conform_name("v", dialect)
    int_type = _int_type(dialect)
    schema = TypedSchema(
        root=[
            Column(name="order", type=SqlType.parse(int_type, dialect)),
            Column(name=v, type=SqlType.parse(int_type, dialect)),
        ]
    )
    expected = TypedResultSet(rows=[{"order": 1, v: 10}], schema=schema)
    model = 'SELECT CAST(1 AS INTEGER) AS "order", CAST(99 AS INTEGER) AS v'
    score = _score(engine, model, expected, _keyed(["order"]), conform_key=False)
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == v


def test_keyed_derived_query_error_fails_without_raise(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "v": 10}), schema=_id_v_schema(dialect))
    score = _score(engine, "SELECT id, v FROM does_not_exist_xyz", expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None


def test_keyed_non_numeric_value_equal_compares_per_column(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    schema = _schema(("id", _int_type(dialect)), ("name", _string_type(dialect)), dialect=dialect)
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "name": "alice"}, {"id": 2, "name": "bob"}), schema=schema)
    model = (
        "SELECT CAST(1 AS INTEGER) AS id, CAST('alice' AS VARCHAR) AS name "
        "UNION ALL SELECT CAST(2 AS INTEGER), CAST('wrong' AS VARCHAR)"
    )
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == conform_name("name", dialect)
    assert score.diff.column_mismatches[0].unexpected_count == 1


def test_keyed_non_numeric_distinct_null_value_is_mismatch(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    schema = _schema(("id", _int_type(dialect)), ("name", _string_type(dialect)), dialect=dialect)
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "name": None}), schema=schema)
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(NULL AS VARCHAR) AS name"
    score = _score(engine, model, expected, _keyed(["id"], null_equality="distinct"))
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == conform_name("name", dialect)


def test_keyed_non_numeric_equal_null_value_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    schema = _schema(("id", _int_type(dialect)), ("name", _string_type(dialect)), dialect=dialect)
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "name": None}), schema=schema)
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(NULL AS VARCHAR) AS name"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is True


def test_keyed_value_columns_named_like_internal_markers(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # Value columns named `e_present`/`a_present` must not collide with the internal presence
    # markers (ambiguous-column on Postgres / silently wrong counts on DuckDB if they did).
    _, dialect = engine
    schema = _schema(
        ("id", _int_type(dialect)),
        ("e_present", _int_type(dialect)),
        ("a_present", _int_type(dialect)),
        dialect=dialect,
    )
    expected = TypedResultSet(rows=_rows(dialect, {"id": 1, "e_present": 10, "a_present": 20}), schema=schema)
    model = "SELECT CAST(1 AS INTEGER) AS id, CAST(99 AS INTEGER) AS e_present, CAST(20 AS INTEGER) AS a_present"
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 0
    assert score.diff.extra_row_count == 0
    assert len(score.diff.column_mismatches) == 1
    assert score.diff.column_mismatches[0].column == conform_name("e_present", dialect)


def test_keyed_multiple_null_keys_not_rejected(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # Two NULL-keyed rows on a side never collide under `=`, so they are not a duplicate key;
    # each surfaces as missing/extra instead of triggering a rejection.
    _, dialect = engine
    expected = TypedResultSet(
        rows=_rows(dialect, {"id": None, "v": 10}, {"id": None, "v": 20}), schema=_id_v_schema(dialect)
    )
    model = (
        "SELECT CAST(NULL AS INTEGER) AS id, CAST(10 AS INTEGER) AS v "
        "UNION ALL SELECT CAST(NULL AS INTEGER), CAST(20 AS INTEGER)"
    )
    score = _score(engine, model, expected, _keyed(["id"]))
    assert score.passed is False
    assert score.diff is not None
    assert score.explanation is None
    assert score.diff.missing_row_count == 2
    assert score.diff.extra_row_count == 2


# ---------- GoldQuery path (the reference query runs in-warehouse) ----------


def test_gold_matches_model(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    expected = GoldQuery(sql="SELECT 1 AS n UNION ALL SELECT 2 AS n")
    score = _score(engine, "SELECT 2 AS n UNION ALL SELECT 1 AS n", expected)
    assert score.passed is True
    assert score.diff is None


def test_gold_differs_from_model(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    expected = GoldQuery(sql="SELECT 1 AS n UNION ALL SELECT 2 AS n")
    score = _score(engine, "SELECT 1 AS n UNION ALL SELECT 3 AS n", expected)
    assert score.passed is False
    assert score.diff is not None
    assert score.diff.missing_row_count == 1
    assert score.diff.extra_row_count == 1
    assert score.diff.sample_missing_rows == _rows(dialect, {"n": 2})
    assert score.diff.sample_extra_rows == _rows(dialect, {"n": 3})


def test_gold_typed_comparison_detects_type_mismatch(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # Gold yields BIGINT, the model yields INTEGER: the gold schema drives a type mismatch.
    _, dialect = engine
    expected = GoldQuery(sql="SELECT CAST(1 AS BIGINT) AS n")
    score = _score(engine, "SELECT CAST(1 AS INTEGER) AS n", expected)
    if dialect in ("sqlite", "snowflake"):
        # SQLite reports no result-column types on either side, so type comparison abstains
        # rather than refutes. Snowflake types both CAST(1 AS BIGINT) and CAST(1 AS INTEGER)
        # identically as NUMBER(38,0), so it can't observe the mismatch either. Either way the
        # rows are otherwise equal, so the score passes.
        assert score.passed is True
        return
    assert score.passed is False
    assert score.diff is not None
    assert len(score.diff.type_mismatches) == 1
    assert score.diff.type_mismatches[0].column == conform_name("n", dialect)


def test_gold_keyed_passes(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    expected = GoldQuery(sql="SELECT 1 AS id, 10 AS v UNION ALL SELECT 2, 20")
    score = _score(engine, "SELECT 2 AS id, 20 AS v UNION ALL SELECT 1, 10", expected, _keyed(["id"]))
    assert score.passed is True
    assert score.diff is None


def test_gold_query_error_is_attributed(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    expected = GoldQuery(sql="SELECT n FROM does_not_exist_xyz")
    score = _score(engine, "SELECT 1 AS n", expected)
    assert score.passed is False
    assert score.diff is None
    assert score.explanation is not None
    assert score.explanation.startswith("gold query failed:")
    assert score.metadata.get("gold_query_failed") is True
