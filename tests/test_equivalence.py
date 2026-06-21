"""Tests for the result-set equivalence engine: column reconciliation and diff assembly."""

import pytest

from evaldata.equivalence import (
    ColumnReconciliation,
    build_result_set_diff,
    combine,
    reconcile_columns,
)
from evaldata.types import ColumnMismatch, ResultSetDiff, SemanticVerdict, TypeMismatch

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


# ---------- combine (semantic-equivalence verdicts) ----------

_SCORER = "semantic_equivalence"


def _verdict(equivalence: str, *, method: str = "ast", diff: ResultSetDiff | None = None) -> SemanticVerdict:
    return SemanticVerdict(method=method, equivalence=equivalence, diff=diff)  # type: ignore[arg-type]


@pytest.mark.unit
class TestCombine:
    def test_no_verdicts_fails_as_undecided(self) -> None:
        score = combine([], scorer=_SCORER)
        assert score.scorer == _SCORER
        assert score.passed is False
        assert score.explanation == "no check could decide equivalence"

    def test_all_unknown_fails_as_undecided(self) -> None:
        score = combine([_verdict("unknown"), _verdict("unknown", method="execution")], scorer=_SCORER)
        assert score.passed is False
        assert score.explanation == "no check could decide equivalence"
        assert len(score.metadata["verdicts"]) == 2

    def test_equivalent_passes_with_no_explanation(self) -> None:
        score = combine([_verdict("equivalent")], scorer=_SCORER)
        assert score.passed is True
        assert score.explanation is None
        assert score.diff is None

    def test_not_equivalent_fails_and_surfaces_diff(self) -> None:
        diff = ResultSetDiff(expected_row_count=1, actual_row_count=0, missing_row_count=1)
        score = combine([_verdict("not_equivalent", method="execution", diff=diff)], scorer=_SCORER)
        assert score.passed is False
        assert score.diff is diff

    def test_first_decisive_verdict_wins_over_later_ones(self) -> None:
        score = combine(
            [_verdict("not_equivalent", method="execution"), _verdict("equivalent")],
            scorer=_SCORER,
        )
        assert score.passed is False

    def test_skips_leading_unknowns_to_first_decision(self) -> None:
        score = combine([_verdict("unknown"), _verdict("equivalent", method="execution")], scorer=_SCORER)
        assert score.passed is True

    def test_records_every_verdict_in_metadata(self) -> None:
        score = combine(
            [_verdict("unknown"), _verdict("unknown", method="plan"), _verdict("equivalent", method="execution")],
            scorer=_SCORER,
        )
        methods = [v["method"] for v in score.metadata["verdicts"]]
        assert methods == ["ast", "plan", "execution"]
