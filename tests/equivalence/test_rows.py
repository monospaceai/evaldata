"""Unit tests for the shared result-set row comparator."""

import pytest

from evaldata.equivalence.rows import compare_rows

pytestmark = pytest.mark.unit


def _cmp(actual, gold, **kwargs):
    return compare_rows(actual, gold, **kwargs)


def test_multiset_is_order_insensitive_but_counts_duplicates() -> None:
    kw = {"order_sensitive": False, "multiplicity": "multiset", "column_alignment": "by_position"}
    assert _cmp([(1,), (2,)], [(2,), (1,)], **kw)
    assert not _cmp([(1,), (1,)], [(1,)], **kw)


def test_set_ignores_duplicates() -> None:
    kw = {"order_sensitive": False, "multiplicity": "set", "column_alignment": "by_position"}
    assert _cmp([(1,), (1,)], [(1,)], **kw)


def test_order_sensitive_requires_same_order() -> None:
    kw = {"order_sensitive": True, "multiplicity": "multiset", "column_alignment": "by_position"}
    assert not _cmp([(1,), (2,)], [(2,), (1,)], **kw)
    assert _cmp([(1,), (2,)], [(1,), (2,)], **kw)


def test_by_value_aligns_permuted_columns() -> None:
    kw = {"order_sensitive": False, "multiplicity": "multiset", "column_alignment": "by_value"}
    assert _cmp([("a", 1)], [(1, "a")], **kw)
    assert not _cmp([("a", 1)], [(1, "b")], **kw)


def test_by_value_handles_empty_and_column_count() -> None:
    kw = {"order_sensitive": False, "multiplicity": "multiset", "column_alignment": "by_value"}
    assert _cmp([], [], **kw)
    assert not _cmp([(1,)], [], **kw)
    assert not _cmp([(1, 2)], [(1,)], **kw)


def test_by_value_wide_results_use_the_pruning_branch() -> None:
    kw = {"order_sensitive": False, "multiplicity": "multiset", "column_alignment": "by_value"}
    assert _cmp([("a", "b", "c", "d")], [("b", "a", "c", "d")], **kw)
    assert not _cmp([("a", "b", "c", "d")], [("b", "a", "c", "e")], **kw)
