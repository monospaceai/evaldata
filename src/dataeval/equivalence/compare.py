"""Pure assembly of a `ResultSetDiff` from precomputed counts, samples, and column signals."""

from typing import Any

from dataeval.equivalence.columns import ColumnReconciliation
from dataeval.types import ColumnMismatch, ResultSetDiff, TypeMismatch


def build_result_set_diff(
    *,
    expected_row_count: int,
    actual_row_count: int,
    missing_row_count: int,
    extra_row_count: int,
    sample_missing_rows: list[dict[str, Any]],
    sample_extra_rows: list[dict[str, Any]],
    columns: ColumnReconciliation,
    type_mismatches: list[TypeMismatch],
    column_mismatches: list[ColumnMismatch],
) -> ResultSetDiff | None:
    """Assemble a `ResultSetDiff` from already-computed diff signals.

    Warehouse-free: the row counts/samples are computed by the engine and the column/type
    signals in Python, then passed here. `column_mismatches` is populated only by the keyed
    `FULL OUTER JOIN` path (empty for the keyless `EXCEPT ALL` path).

    Args:
        expected_row_count: The number of expected rows.
        actual_row_count: The number of actual rows.
        missing_row_count: Rows present in expected but absent from actual.
        extra_row_count: Rows present in actual but absent from expected.
        sample_missing_rows: A bounded sample of the missing rows.
        sample_extra_rows: A bounded sample of the extra rows.
        columns: The reconciliation of actual against expected column names.
        type_mismatches: Per-column type differences over the shared columns.
        column_mismatches: Per-column counts of key-matched rows whose value differs;
            empty for the keyless path.

    Returns:
        `None` if the assembled diff records no differences (the result sets are equal),
        else the populated `ResultSetDiff`.
    """
    diff = ResultSetDiff(
        expected_row_count=expected_row_count,
        actual_row_count=actual_row_count,
        missing_row_count=missing_row_count,
        extra_row_count=extra_row_count,
        sample_missing_rows=sample_missing_rows,
        sample_extra_rows=sample_extra_rows,
        missing_columns=columns.missing,
        unexpected_columns=columns.unexpected,
        type_mismatches=type_mismatches,
        column_mismatches=column_mismatches,
        column_order_mismatch=columns.order_mismatch,
    )
    if _is_equal(diff):
        return None
    return diff


def _is_equal(d: ResultSetDiff) -> bool:
    """Whether the diff records no differences — i.e. the result sets are equal.

    Args:
        d: The diff to inspect.

    Returns:
        `True` if there are no row, column, type, or ordering differences.
    """
    return (
        d.missing_row_count == 0
        and d.extra_row_count == 0
        and not d.missing_columns
        and not d.unexpected_columns
        and not d.type_mismatches
        and not d.column_mismatches
        and not d.column_order_mismatch
    )
