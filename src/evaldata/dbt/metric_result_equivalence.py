"""`MetricResultEquivalence`: decide equivalence by running both metric queries and diffing rows."""

import itertools
from typing import Literal

from evaldata.dbt.errors import DbtError
from evaldata.dbt.metricflow import run
from evaldata.dbt.semantic_layer import MetricCase, MetricQuery
from evaldata.equivalence.rows import Row, compare_rows
from evaldata.types import ScoreResult

SCORER_NAME = "metric_result_equivalence"

# Numeric cells are compared after rounding to this many decimals, so CSV formatting (an integer
# versus a trailing-zero float) does not read as a difference.
_ROUND = 6


class MetricResultEquivalence:
    """Decides equivalence by running both metric queries and comparing their result rows.

    Rows are compared as an order-insensitive multiset. Columns are aligned by value, so the same
    answer under a different metric or dimension label still matches, and numeric cells match within
    a small tolerance. A candidate that groups by extra columns still matches when they are
    redundant (dropping them keeps every row distinct). A model-query run failure is inconclusive,
    or failing under `on_error="fail"`; a gold-query run failure is always inconclusive.
    """

    def __init__(self, *, on_error: Literal["inconclusive", "fail"] = "inconclusive") -> None:
        """Configure how a failed model-query run is scored.

        Args:
            on_error: `"inconclusive"` (default) to defer a failed model query to a later tier, or
                `"fail"` to score it as incorrect.
        """
        self._on_error = on_error

    def score(self, case: MetricCase, query: MetricQuery) -> ScoreResult:
        """Run both queries and decide equivalence from their result rows.

        Args:
            case: The eval case, supplying the gold query and the target directory.
            query: The candidate metric query.

        Returns:
            A passing or failing, observed `ScoreResult` when both queries run, else an
            inconclusive result (or a failing one for the model query under `on_error="fail"`).
        """
        candidate = run(query, case.target_dir, profiles_dir=case.profiles_dir)
        if isinstance(candidate, DbtError):
            if self._on_error == "fail":
                return ScoreResult(
                    scorer=SCORER_NAME,
                    verdict="fail",
                    basis="observed",
                    explanation=f"model query failed: {candidate.message}",
                )
            return _inconclusive(f"model query: {candidate.message}")
        gold = run(case.gold, case.target_dir, profiles_dir=case.profiles_dir)
        if isinstance(gold, DbtError):
            return _inconclusive(f"gold query: {gold.message}")

        if _rows_match(_tuples(candidate), _tuples(gold)):
            return ScoreResult(
                scorer=SCORER_NAME,
                verdict="pass",
                basis="observed",
                explanation="metric queries return the same rows",
            )
        return ScoreResult(
            scorer=SCORER_NAME,
            verdict="fail",
            basis="observed",
            explanation=f"metric queries differ: gold returned {len(gold)} rows, model returned {len(candidate)}",
        )


def _cell(value: str) -> float | str:
    try:
        return round(float(value), _ROUND)
    except ValueError:
        return value


def _tuples(rows: list[dict[str, str]]) -> list[Row]:
    return [tuple(_cell(value) for value in row.values()) for row in rows]


def _rows_match(candidate: list[Row], gold: list[Row]) -> bool:
    # Equal rows, but a candidate may group by extra columns when they are redundant.
    extra = (len(candidate[0]) - len(gold[0])) if candidate and gold else 0
    if extra <= 0:
        return _equal(candidate, gold)
    # The candidate groups by more columns; the extra grouping is redundant only when some
    # projection onto the gold's column count keeps every candidate row distinct.
    distinct = len(set(candidate))
    for columns in itertools.combinations(range(len(candidate[0])), len(gold[0])):
        projected = [tuple(row[index] for index in columns) for row in candidate]
        if len(set(projected)) == distinct and _equal(projected, gold):
            return True
    return False


def _equal(candidate: list[Row], gold: list[Row]) -> bool:
    return compare_rows(candidate, gold, order_sensitive=False, multiplicity="multiset", column_alignment="by_value")


def _inconclusive(detail: str) -> ScoreResult:
    return ScoreResult(scorer=SCORER_NAME, verdict="inconclusive", explanation=detail)
