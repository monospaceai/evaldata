"""Public `compare()` entry: dispatches on typed-vs-untyped result sets, returning a `ResultSetDiff` or `None`."""

from typing import overload

from data_eval.equivalence.columns import ColumnReconciliation, reconcile_columns
from data_eval.equivalence.result_set import TypedResultSet, UntypedResultSet
from data_eval.equivalence.rows import match_multiset
from data_eval.equivalence.types import types_match
from data_eval.types import ComparisonConfig, ResultSetDiff, SQLDialect, TypeMismatch

#: Max differing rows carried in each `ResultSetDiff` sample. Counts stay exact; only
#: the inline examples are capped so a large mismatch doesn't flood the failure message.
SAMPLE_LIMIT = 10


@overload
def compare(
    actual: TypedResultSet,
    expected: TypedResultSet,
    config: ComparisonConfig | None = ...,
    *,
    compare_types: bool = ...,
    dialect: SQLDialect | None = ...,
) -> ResultSetDiff | None: ...


@overload
def compare(
    actual: UntypedResultSet,
    expected: UntypedResultSet,
    config: ComparisonConfig | None = ...,
) -> ResultSetDiff | None: ...


def compare(
    actual: TypedResultSet | UntypedResultSet,
    expected: TypedResultSet | UntypedResultSet,
    config: ComparisonConfig | None = None,
    *,
    compare_types: bool = True,
    dialect: SQLDialect | None = None,
) -> ResultSetDiff | None:
    """Compare an actual result set against an expected one for equivalence.

    Both inputs must be the same shape: either both `TypedResultSet` (enabling semantic
    type comparison) or both `UntypedResultSet`. Row comparison is multiset (unordered).

    Args:
        actual: The actual result set.
        expected: The expected result set.
        config: Equivalence rules (column ordering, null equality, float tolerance).
            Defaults to `ComparisonConfig()`.
        compare_types: Whether to compare column types. Only meaningful for typed result
            sets.
        dialect: The SQLGlot dialect for semantic type comparison. Required when
            `compare_types` is true on typed result sets.

    Returns:
        `None` if the result sets are equivalent, else a `ResultSetDiff` describing the
        differences.

    Raises:
        TypeError: If `actual` and `expected` are not both typed or both untyped.
    """
    cfg = config or ComparisonConfig()
    if isinstance(actual, TypedResultSet) and isinstance(expected, TypedResultSet):
        return _compare_typed(actual, expected, cfg, compare_types, dialect)
    if isinstance(actual, UntypedResultSet) and isinstance(expected, UntypedResultSet):
        return _compare_untyped(actual, expected, cfg)
    msg = "actual and expected must both be Typed or both Untyped result sets"
    raise TypeError(msg)


def _compare_typed(
    actual: TypedResultSet,
    expected: TypedResultSet,
    cfg: ComparisonConfig,
    compare_types: bool,
    dialect: SQLDialect | None,
) -> ResultSetDiff | None:
    """Compare two typed result sets, optionally including semantic column-type comparison.

    Returns:
        `None` if equivalent, else a `ResultSetDiff` describing the differences.

    Raises:
        ValueError: If `compare_types` is true but `dialect` is not given.
    """
    cols = reconcile_columns(
        [c.name for c in actual.schema_],
        [c.name for c in expected.schema_],
        cfg.column_order,
    )
    type_mismatches: list[TypeMismatch] = []
    if compare_types:
        if dialect is None:
            msg = "`dialect` is required when `compare_types=True` on typed result sets"
            raise ValueError(msg)
        actual_types = {c.name: c.type for c in actual.schema_}
        expected_types = {c.name: c.type for c in expected.schema_}
        type_mismatches = [
            TypeMismatch(column=col, expected=expected_types[col], actual=actual_types[col])
            for col in cols.in_both
            if not types_match(actual_types[col], expected_types[col], dialect)
        ]
    return _build_diff(actual, expected, cols, type_mismatches, cfg)


def _compare_untyped(
    actual: UntypedResultSet,
    expected: UntypedResultSet,
    cfg: ComparisonConfig,
) -> ResultSetDiff | None:
    """Compare two untyped result sets; column names come from the first row of each.

    Returns:
        `None` if equivalent, else a `ResultSetDiff` describing the differences.
    """
    cols = reconcile_columns(
        list(actual.rows[0].keys()) if actual.rows else [],
        list(expected.rows[0].keys()) if expected.rows else [],
        cfg.column_order,
    )
    return _build_diff(actual, expected, cols, [], cfg)


def _build_diff(
    actual: TypedResultSet | UntypedResultSet,
    expected: TypedResultSet | UntypedResultSet,
    cols: ColumnReconciliation,
    type_mismatches: list[TypeMismatch],
    cfg: ComparisonConfig,
) -> ResultSetDiff | None:
    """Assemble the row-level diff and merge in the already-computed column/type signals.

    Returns:
        `None` if the assembled diff carries no differences, else the `ResultSetDiff`.
    """
    missing_rows, extra_rows = match_multiset(
        actual.rows,
        expected.rows,
        cols.in_both,
        cfg.null_equality,
        cfg.float_tolerance,
    )
    diff = ResultSetDiff(
        expected_row_count=len(expected.rows),
        actual_row_count=len(actual.rows),
        missing_row_count=len(missing_rows),
        extra_row_count=len(extra_rows),
        sample_missing_rows=missing_rows[:SAMPLE_LIMIT],
        sample_extra_rows=extra_rows[:SAMPLE_LIMIT],
        missing_columns=cols.missing,
        unexpected_columns=cols.unexpected,
        type_mismatches=type_mismatches,
        column_order_mismatch=cols.order_mismatch,
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
