"""Column reconciliation between actual and expected schemas."""

from typing import Literal, NamedTuple


class ColumnReconciliation(NamedTuple):
    """The outcome of reconciling actual against expected column names.

    Attributes:
        in_both: Columns present in both, in expected order; the columns compared on.
        missing: Columns expected but absent from actual, in expected order.
        unexpected: Columns present in actual but not expected, in actual order.
        order_mismatch: `True` only when `column_order == "strict"` and the sequences
            differ positionally.
    """

    in_both: list[str]
    missing: list[str]
    unexpected: list[str]
    order_mismatch: bool


def reconcile_columns(
    actual: list[str],
    expected: list[str],
    column_order: Literal["ignore", "strict"],
) -> ColumnReconciliation:
    """Reconcile actual against expected column-name sequences.

    Row comparison is always keyed by name (rows are dicts), so the order signal is a
    separate assertion rather than a constraint on row matching.

    Args:
        actual: Column names from the actual result set.
        expected: Column names from the expected result set.
        column_order: `"strict"` to flag a positional order difference, `"ignore"` to
            disregard ordering.

    Returns:
        A `ColumnReconciliation`. The `in_both`/`missing`/`unexpected` lists preserve
        source order by construction (the sets are membership lookups only).
    """
    actual_set = set(actual)
    expected_set = set(expected)
    in_both = [c for c in expected if c in actual_set]
    missing = [c for c in expected if c not in actual_set]
    unexpected = [c for c in actual if c not in expected_set]
    order_mismatch = column_order == "strict" and actual != expected
    return ColumnReconciliation(in_both, missing, unexpected, order_mismatch)
