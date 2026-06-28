"""Property/fuzz parity: `ExecutionAccuracy._compare` vs the official Spider/BIRD oracles.

Isolates the comparison logic from SQL execution and order detection: random result sets are
fed straight to `_compare` and to the official comparators, asserting they agree across many
seeded cases. Rows are tuples of hashable cells drawn from a tiny domain so duplicates, NULLs,
and column collisions are common.

- Spider config (`column_alignment="by_value"`) is checked against the vendored `result_eq`.
- BIRD config (`row_order="ignore", multiplicity="set"`) is checked against `set == set`.

Argument order matches `test_official_parity.py`: our `_compare(actual=pred, gold=gold)` and the
oracle `result_eq(result1=pred, result2=gold)`.
"""

import random

import pytest

from evaldata.scorers import ExecutionAccuracy
from tests._vendor.spider_exec_eval import result_eq

_SEED = 1234
_ITERATIONS = 2000
_DOMAIN: tuple[object, ...] = (0, 1, "x", "y", None)
_Tuple = tuple[object, ...]


def _random_rows(rng: random.Random, num_cols: int, num_rows: int) -> list[_Tuple]:
    """Build `num_rows` tuples of `num_cols` cells drawn from the tiny shared domain."""
    return [tuple(rng.choice(_DOMAIN) for _ in range(num_cols)) for _ in range(num_rows)]


def _mutate(rng: random.Random, gold: list[_Tuple]) -> list[_Tuple]:
    """Apply one random transform to `gold` to derive a candidate prediction.

    The transforms span the edge cases that separate the oracles: row order, duplicates,
    column order, column count, and per-cell or whole-set differences.
    """
    transform = rng.choice(
        ("identity", "shuffle", "dedup", "permute_cols", "drop_col", "add_col", "mutate_cell", "empty")
    )
    if transform == "identity" or not gold:
        return list(gold)
    if transform == "shuffle":
        rows = list(gold)
        rng.shuffle(rows)
        return rows
    if transform == "dedup":
        seen: set[_Tuple] = set()
        out: list[_Tuple] = []
        for row in gold:
            if row not in seen:
                seen.add(row)
                out.append(row)
        return out
    num_cols = len(gold[0])
    if transform == "permute_cols" and num_cols > 1:
        perm = list(range(num_cols))
        rng.shuffle(perm)
        return [tuple(row[i] for i in perm) for row in gold]
    if transform == "drop_col" and num_cols > 1:
        drop = rng.randrange(num_cols)
        return [tuple(v for i, v in enumerate(row) if i != drop) for row in gold]
    if transform == "add_col":
        return [(*row, rng.choice(_DOMAIN)) for row in gold]
    if transform == "mutate_cell":
        rows = [list(row) for row in gold]
        r = rng.randrange(len(rows))
        c = rng.randrange(num_cols)
        rows[r][c] = rng.choice(_DOMAIN)
        return [tuple(row) for row in rows]
    if transform == "empty":
        return []
    return list(gold)


@pytest.mark.unit
def test_compare_matches_oracles_over_random_inputs() -> None:
    """`_compare` agrees with the Spider and BIRD oracles across {_ITERATIONS} seeded cases."""
    rng = random.Random(_SEED)
    spider = ExecutionAccuracy(column_alignment="by_value")
    bird = ExecutionAccuracy(row_order="ignore", multiplicity="set")
    for _ in range(_ITERATIONS):
        num_cols = rng.randint(1, 4)
        num_rows = rng.randint(0, 6)
        gold = _random_rows(rng, num_cols, num_rows)
        pred = _mutate(rng, gold)
        order_sensitive = rng.choice((True, False))

        mine = spider._compare(pred, gold, order_sensitive)
        official = result_eq(pred, gold, order_matters=order_sensitive)
        assert mine == official, (
            f"Spider mismatch: pred={pred!r} gold={gold!r} order_sensitive={order_sensitive!r} "
            f'config="by_value" mine={mine!r} official={official!r}'
        )

        mine_bird = bird._compare(pred, gold, order_sensitive=False)
        official_bird = set(pred) == set(gold)
        assert mine_bird == official_bird, (
            f"BIRD mismatch: pred={pred!r} gold={gold!r} "
            f'config="set/ignore" mine={mine_bird!r} official={official_bird!r}'
        )
