"""Pure result-set row comparison: row order, duplicate handling, and by-value column alignment."""

import itertools
from collections import Counter
from typing import Any, Literal

Row = tuple[Any, ...]

Multiplicity = Literal["multiset", "set"]
ColumnAlignment = Literal["by_position", "by_value"]

# Above this column count, permutation candidates are pruned against sampled rows rather than
# enumerating the full cartesian product.
_FULL_PRODUCT_MAX_COLS = 3
_PRUNE_SAMPLE_ROWS = 20


def compare_rows(
    actual: list[Row],
    gold: list[Row],
    *,
    order_sensitive: bool,
    multiplicity: Multiplicity,
    column_alignment: ColumnAlignment,
) -> bool:
    """Whether two result sets are equal under the given order, duplicate, and column semantics.

    Args:
        actual: The model result rows as positional tuples.
        gold: The gold result rows as positional tuples.
        order_sensitive: Whether row order must match.
        multiplicity: `"multiset"` to compare with duplicate counts, `"set"` for distinct rows.
        column_alignment: `"by_position"` to compare columns in order, `"by_value"` to accept any
            column permutation whose values make the sets match (requires equal column counts).

    Returns:
        Whether the result sets are considered equal.
    """
    if column_alignment == "by_value":
        return _compare_by_value(actual, gold, order_sensitive, multiplicity)
    return _compare_positional(actual, gold, order_sensitive, multiplicity)


def _compare_positional(actual: list[Row], gold: list[Row], order_sensitive: bool, multiplicity: Multiplicity) -> bool:
    if order_sensitive:
        return actual == gold
    if multiplicity == "set":
        return set(actual) == set(gold)
    return Counter(actual) == Counter(gold)


def _compare_by_value(actual: list[Row], gold: list[Row], order_sensitive: bool, multiplicity: Multiplicity) -> bool:
    if not actual and not gold:
        return True
    if not actual or not gold:
        return False
    num_cols = len(actual[0])
    if len(gold[0]) != num_cols:
        return False
    for perm in _column_permutations(actual, gold, num_cols):
        if len(set(perm)) != len(perm):
            continue
        gold_permuted = [tuple(row[i] for i in perm) for row in gold]
        if _compare_positional(actual, gold_permuted, order_sensitive, multiplicity):
            return True
    return False


def _column_permutations(actual: list[Row], gold: list[Row], num_cols: int) -> list[tuple[int, ...]]:
    """Candidate gold-column permutations to test against the model columns.

    Args:
        actual: The model result rows as positional tuples.
        gold: The gold result rows as positional tuples.
        num_cols: The shared column count.

    Returns:
        Candidate permutations mapping each target position to a source gold column. For narrow
        results every mapping is enumerated; for wider results the per-position candidates are
        pruned against sampled values first.
    """
    if num_cols <= _FULL_PRODUCT_MAX_COLS:
        return list(itertools.product(range(num_cols), repeat=num_cols))

    gold_value_sets = [{row[i] for row in gold} for i in range(num_cols)]
    candidates: list[set[int]] = [set(range(num_cols)) for _ in range(num_cols)]
    # Iterate the first rows rather than sampling at random so eval runs stay reproducible; full
    # equality is still verified per surviving permutation.
    for row in actual[:_PRUNE_SAMPLE_ROWS]:
        for target, value in enumerate(row):
            candidates[target] = {src for src in candidates[target] if value in gold_value_sets[src]}
    return list(itertools.product(*candidates))
