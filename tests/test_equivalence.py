"""Tests for the result-set equivalence engine: input types, reconciliation, diff assembly."""

import pytest
from pydantic import ValidationError

from data_eval.equivalence import (
    ColumnReconciliation,
    TypedResultSet,
    UntypedResultSet,
    build_result_set_diff,
    reconcile_columns,
)
from data_eval.types import Column, ColumnMismatch, TypeMismatch

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


# ---------- column reconciliation ----------


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


# ---------- build_result_set_diff ----------


def _cols(
    in_both: list[str],
    *,
    missing: list[str] | None = None,
    unexpected: list[str] | None = None,
    order_mismatch: bool = False,
) -> ColumnReconciliation:
    return ColumnReconciliation(in_both, missing or [], unexpected or [], order_mismatch)


def _diff(**overrides: object) -> object:
    kwargs: dict[str, object] = {
        "expected_row_count": 1,
        "actual_row_count": 1,
        "missing_row_count": 0,
        "extra_row_count": 0,
        "sample_missing_rows": [],
        "sample_extra_rows": [],
        "columns": _cols(["n"]),
        "type_mismatches": [],
        "column_mismatches": [],
    }
    kwargs.update(overrides)
    return build_result_set_diff(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
class TestBuildResultSetDiff:
    def test_equal_returns_none(self) -> None:
        assert _diff() is None

    def test_missing_rows_flag_difference(self) -> None:
        diff = _diff(missing_row_count=1, sample_missing_rows=[{"n": 2}])
        assert diff is not None
        assert diff.missing_row_count == 1
        assert diff.sample_missing_rows == [{"n": 2}]

    def test_extra_rows_flag_difference(self) -> None:
        diff = _diff(extra_row_count=1, sample_extra_rows=[{"n": 2}])
        assert diff is not None
        assert diff.extra_row_count == 1
        assert diff.sample_extra_rows == [{"n": 2}]

    def test_missing_columns_flag_difference(self) -> None:
        diff = _diff(columns=_cols(["a"], missing=["b"]))
        assert diff is not None
        assert diff.missing_columns == ["b"]

    def test_unexpected_columns_flag_difference(self) -> None:
        diff = _diff(columns=_cols(["a"], unexpected=["c"]))
        assert diff is not None
        assert diff.unexpected_columns == ["c"]

    def test_column_order_mismatch_flags_difference(self) -> None:
        diff = _diff(columns=_cols(["a", "b"], order_mismatch=True))
        assert diff is not None
        assert diff.column_order_mismatch is True

    def test_type_mismatch_flags_difference(self) -> None:
        diff = _diff(type_mismatches=[TypeMismatch(column="n", expected="INT", actual="BIGINT")])
        assert diff is not None
        assert len(diff.type_mismatches) == 1
        assert diff.type_mismatches[0].column == "n"

    def test_column_mismatches_default_empty(self) -> None:
        diff = _diff(extra_row_count=1, sample_extra_rows=[{"n": 2}])
        assert diff is not None
        assert diff.column_mismatches == []

    def test_column_mismatches_flag_difference(self) -> None:
        diff = _diff(column_mismatches=[ColumnMismatch(column="n", unexpected_count=2)])
        assert diff is not None
        assert len(diff.column_mismatches) == 1
        assert diff.column_mismatches[0].column == "n"
        assert diff.column_mismatches[0].unexpected_count == 2

    def test_carries_row_counts(self) -> None:
        diff = _diff(expected_row_count=3, actual_row_count=5, extra_row_count=2, sample_extra_rows=[{"n": 9}])
        assert diff is not None
        assert diff.expected_row_count == 3
        assert diff.actual_row_count == 5
