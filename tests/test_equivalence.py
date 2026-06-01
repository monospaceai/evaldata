"""Tests for the result-set equivalence engine."""

from typing import Any, cast

import pytest
from pydantic import ValidationError

from data_eval.equivalence import TypedResultSet, UntypedResultSet, compare
from data_eval.equivalence.columns import reconcile_columns
from data_eval.equivalence.rows import match_multiset
from data_eval.equivalence.values import cells_equal
from data_eval.types import Column, ComparisonConfig, SqlType

# ---------- engine input types ----------


@pytest.mark.unit
class TestUntypedResultSet:
    def test_empty_construction(self) -> None:
        rs = UntypedResultSet(rows=[])
        assert rs.rows == []

    def test_with_rows(self) -> None:
        rs = UntypedResultSet(rows=[{"x": 1}, {"x": 2}])
        assert len(rs.rows) == 2

    def test_json_round_trip(self) -> None:
        rs = UntypedResultSet(rows=[{"x": 1}])
        restored = UntypedResultSet.model_validate_json(rs.model_dump_json())
        assert restored == rs

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            UntypedResultSet.model_validate({"rows": [], "schema": []})


@pytest.mark.unit
class TestTypedResultSet:
    def test_minimal_construction(self) -> None:
        rs = TypedResultSet(rows=[], schema=[Column(name="id", type="INTEGER")])
        assert rs.rows == []
        assert rs.schema_.names == ["id"]
        assert rs.schema_[0].type.raw == "INTEGER"

    def test_with_rows_and_schema(self) -> None:
        rs = TypedResultSet(
            rows=[{"id": 1, "name": "rock"}],
            schema=[Column(name="id", type="BIGINT"), Column(name="name", type="VARCHAR")],
        )
        assert len(rs.schema_) == 2

    def test_nested_type_in_schema(self) -> None:
        rs = TypedResultSet(
            rows=[{"payload": [{"a": 1}]}],
            schema=[Column(name="payload", type="ARRAY<STRUCT<a: INT>>")],
        )
        assert rs.schema_[0].type.raw == "ARRAY<STRUCT<a: INT>>"

    def test_json_round_trip_uses_external_alias(self) -> None:
        rs = TypedResultSet(rows=[{"id": 1}], schema=[Column(name="id", type="INTEGER")])
        dumped = rs.model_dump_json()
        assert '"schema"' in dumped
        assert '"schema_"' not in dumped
        restored = TypedResultSet.model_validate_json(dumped)
        assert restored == rs

    def test_schema_required(self) -> None:
        with pytest.raises(ValidationError):
            TypedResultSet.model_validate({"rows": []})

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            TypedResultSet.model_validate(
                {"rows": [], "schema": [{"name": "x", "type": "INT"}], "dialect": "duckdb"},
            )


# ---------- helper-function units ----------


@pytest.mark.unit
class TestReconcileColumns:
    def test_identical_columns_ignore(self) -> None:
        rec = reconcile_columns(["a", "b"], ["a", "b"], "ignore")
        assert rec.in_both == ["a", "b"]
        assert rec.missing == []
        assert rec.unexpected == []
        assert rec.order_mismatch is False

    def test_set_difference(self) -> None:
        rec = reconcile_columns(["a", "c"], ["a", "b"], "ignore")
        assert rec.in_both == ["a"]
        assert rec.missing == ["b"]
        assert rec.unexpected == ["c"]

    def test_strict_flags_positional_mismatch_with_equal_sets(self) -> None:
        rec = reconcile_columns(["b", "a"], ["a", "b"], "strict")
        assert rec.in_both == ["a", "b"]
        assert rec.missing == []
        assert rec.unexpected == []
        assert rec.order_mismatch is True

    def test_ignore_does_not_flag_order(self) -> None:
        rec = reconcile_columns(["b", "a"], ["a", "b"], "ignore")
        assert rec.order_mismatch is False


@pytest.mark.unit
class TestCellsEqual:
    def test_both_null_equal(self) -> None:
        assert cells_equal(None, None, "equal", 0.0) is True

    def test_both_null_distinct(self) -> None:
        assert cells_equal(None, None, "distinct", 0.0) is False

    def test_one_null(self) -> None:
        assert cells_equal(None, 1, "equal", 0.0) is False
        assert cells_equal(1, None, "equal", 0.0) is False

    def test_numeric_within_tolerance(self) -> None:
        assert cells_equal(1.0, 1.0 + 1e-10, "equal", 1e-9) is True

    def test_numeric_outside_tolerance(self) -> None:
        assert cells_equal(1.0, 1.1, "equal", 1e-9) is False

    def test_int_vs_float_equal(self) -> None:
        assert cells_equal(42, 42.0, "equal", 0.0) is True

    def test_strings_equal(self) -> None:
        assert cells_equal("rock", "rock", "equal", 0.0) is True
        assert cells_equal("rock", "pop", "equal", 0.0) is False


@pytest.mark.unit
class TestMatchMultiset:
    def test_identical_lists(self) -> None:
        missing, extra = match_multiset(
            [{"x": 1}, {"x": 2}],
            [{"x": 1}, {"x": 2}],
            ["x"],
            "equal",
            0.0,
        )
        assert (missing, extra) == ([], [])

    def test_different_order_still_matches(self) -> None:
        missing, extra = match_multiset(
            [{"x": 2}, {"x": 1}],
            [{"x": 1}, {"x": 2}],
            ["x"],
            "equal",
            0.0,
        )
        assert (missing, extra) == ([], [])

    def test_duplicates_treated_as_multiset(self) -> None:
        # [{1},{1}] vs [{1}] reports one extra (bag semantics; a set would say equivalent)
        missing, extra = match_multiset(
            [{"x": 1}, {"x": 1}],
            [{"x": 1}],
            ["x"],
            "equal",
            0.0,
        )
        assert missing == []
        assert extra == [{"x": 1}]

    def test_missing_row(self) -> None:
        missing, extra = match_multiset(
            [{"x": 1}],
            [{"x": 1}, {"x": 2}],
            ["x"],
            "equal",
            0.0,
        )
        assert missing == [{"x": 2}]
        assert extra == []


# ---------- compare() battery ----------


def _typed(
    rows: list[dict[str, Any]],
    names: list[str],
    types: list[str],
    dialect: str = "duckdb",
) -> TypedResultSet:
    cols = [Column(name=n, type=SqlType.parse(t, dialect)) for n, t in zip(names, types, strict=True)]  # ty: ignore[invalid-argument-type]
    return TypedResultSet(rows=rows, schema=cols)


def _untyped(rows: list[dict[str, Any]]) -> UntypedResultSet:
    return UntypedResultSet(rows=rows)


@pytest.mark.unit
class TestCompareIdentity:
    def test_identical_untyped(self) -> None:
        assert compare(_untyped([{"n": 1}]), _untyped([{"n": 1}])) is None

    def test_identical_typed(self) -> None:
        a = _typed([{"n": 1}], ["n"], ["INTEGER"])
        b = _typed([{"n": 1}], ["n"], ["INTEGER"])
        assert compare(a, b) is None

    def test_value_mismatch(self) -> None:
        diff = compare(_untyped([{"n": 1}]), _untyped([{"n": 2}]))
        assert diff is not None
        assert diff.missing_row_count == 1
        assert diff.extra_row_count == 1


@pytest.mark.unit
class TestCompareColumnOrder:
    def test_ignore_order_passes(self) -> None:
        # rows are dict-keyed; column reordering shouldn't fail by default
        a = _untyped([{"a": 1, "b": 2}])
        b = _untyped([{"b": 2, "a": 1}])
        assert compare(a, b) is None

    def test_strict_order_mismatch_flags(self) -> None:
        a = _typed([{"a": 1, "b": 2}], ["a", "b"], ["INT", "INT"])
        b = _typed([{"a": 1, "b": 2}], ["b", "a"], ["INT", "INT"])
        diff = compare(a, b, ComparisonConfig(column_order="strict"))
        assert diff is not None
        assert diff.column_order_mismatch is True

    def test_strict_order_match_passes(self) -> None:
        a = _typed([{"a": 1, "b": 2}], ["a", "b"], ["INT", "INT"])
        b = _typed([{"a": 1, "b": 2}], ["a", "b"], ["INT", "INT"])
        assert compare(a, b, ComparisonConfig(column_order="strict")) is None

    def test_missing_columns(self) -> None:
        a = _untyped([{"a": 1}])
        b = _untyped([{"a": 1, "b": 2}])
        diff = compare(a, b)
        assert diff is not None
        assert diff.missing_columns == ["b"]

    def test_unexpected_columns(self) -> None:
        a = _untyped([{"a": 1, "b": 2}])
        b = _untyped([{"a": 1}])
        diff = compare(a, b)
        assert diff is not None
        assert diff.unexpected_columns == ["b"]


@pytest.mark.unit
class TestCompareNullEquality:
    def test_null_equal_default(self) -> None:
        assert compare(_untyped([{"n": None}]), _untyped([{"n": None}])) is None

    def test_null_distinct_flags_mismatch(self) -> None:
        diff = compare(
            _untyped([{"n": None}]),
            _untyped([{"n": None}]),
            ComparisonConfig(null_equality="distinct"),
        )
        assert diff is not None
        assert diff.missing_row_count == 1


@pytest.mark.unit
class TestCompareFloatTolerance:
    def test_inside_default_tolerance(self) -> None:
        # default tol is 1e-9; 1e-10 is well within
        assert compare(_untyped([{"v": 1.0 + 1e-10}]), _untyped([{"v": 1.0}])) is None

    def test_outside_default_tolerance(self) -> None:
        diff = compare(_untyped([{"v": 1.0 + 1e-6}]), _untyped([{"v": 1.0}]))
        assert diff is not None
        assert diff.missing_row_count == 1

    def test_tighter_tolerance_rejects(self) -> None:
        diff = compare(
            _untyped([{"v": 1.0 + 1e-10}]),
            _untyped([{"v": 1.0}]),
            ComparisonConfig(float_tolerance=1e-12),
        )
        assert diff is not None


@pytest.mark.unit
class TestCompareMultiset:
    def test_duplicates_not_collapsed_to_set(self) -> None:
        # bag semantics: [{1},{1}] vs [{1}] flags one extra; set semantics would say equivalent
        diff = compare(_untyped([{"x": 1}, {"x": 1}]), _untyped([{"x": 1}]))
        assert diff is not None
        assert diff.extra_row_count == 1
        assert diff.missing_row_count == 0

    def test_unordered_rows_match(self) -> None:
        a = _untyped([{"x": 2}, {"x": 1}, {"x": 3}])
        b = _untyped([{"x": 1}, {"x": 2}, {"x": 3}])
        assert compare(a, b) is None

    def test_diff_carries_differing_row_samples(self) -> None:
        # actual has an unexpected row; expected has one that's absent — both surface as samples
        diff = compare(_untyped([{"n": 1298}]), _untyped([{"n": 1297}]))
        assert diff is not None
        assert diff.sample_extra_rows == [{"n": 1298}]
        assert diff.sample_missing_rows == [{"n": 1297}]

    def test_row_samples_are_capped(self) -> None:
        actual = _untyped([{"n": i} for i in range(100, 130)])
        expected = _untyped([])
        diff = compare(actual, expected)
        assert diff is not None
        assert diff.extra_row_count == 30  # full magnitude
        assert len(diff.sample_extra_rows) == 10  # SAMPLE_LIMIT


@pytest.mark.unit
class TestCompareTypes:
    def test_bigint_long_databricks_equivalent(self) -> None:
        a = _typed([{"n": 1}], ["n"], ["BIGINT"], dialect="databricks")
        b = _typed([{"n": 1}], ["n"], ["LONG"], dialect="databricks")
        assert compare(a, b) is None

    def test_bigint_int8_duckdb_equivalent(self) -> None:
        a = _typed([{"n": 1}], ["n"], ["BIGINT"])
        b = _typed([{"n": 1}], ["n"], ["INT8"])
        assert compare(a, b) is None

    def test_integer_vs_bigint_distinct(self) -> None:
        a = _typed([{"n": 1}], ["n"], ["INTEGER"])
        b = _typed([{"n": 1}], ["n"], ["BIGINT"])
        diff = compare(a, b)
        assert diff is not None
        assert len(diff.type_mismatches) == 1
        assert diff.type_mismatches[0].column == "n"

    def test_compare_types_false_skips_type_check(self) -> None:
        a = _typed([{"n": 1}], ["n"], ["INTEGER"])
        b = _typed([{"n": 1}], ["n"], ["BIGINT"])
        assert compare(a, b, compare_types=False) is None

    def test_nested_types_compared_structurally(self) -> None:
        a = _typed([{"p": [{"a": 1}]}], ["p"], ["ARRAY<STRUCT<a: INT>>"], dialect="databricks")
        b = _typed([{"p": [{"a": 1}]}], ["p"], ["ARRAY<STRUCT<a: STRING>>"], dialect="databricks")
        diff = compare(a, b)
        assert diff is not None
        assert len(diff.type_mismatches) == 1


@pytest.mark.unit
class TestCompareEdgeCases:
    def test_both_empty(self) -> None:
        assert compare(_untyped([]), _untyped([])) is None

    def test_columns_differ_only(self) -> None:
        a = _untyped([{"a": 1}])
        b = _untyped([{"b": 1}])
        diff = compare(a, b)
        assert diff is not None
        assert diff.missing_columns == ["b"]
        assert diff.unexpected_columns == ["a"]

    def test_types_differ_only(self) -> None:
        a = _typed([{"n": 1}], ["n"], ["INTEGER"])
        b = _typed([{"n": 1}], ["n"], ["BIGINT"])
        diff = compare(a, b)
        assert diff is not None
        assert len(diff.type_mismatches) == 1
        assert diff.missing_row_count == 0
        assert diff.extra_row_count == 0

    def test_mixed_typed_untyped_raises(self) -> None:
        typed = _typed([{"n": 1}], ["n"], ["INTEGER"])
        untyped = _untyped([{"n": 1}])
        with pytest.raises(TypeError):
            cast(Any, compare)(typed, untyped)
