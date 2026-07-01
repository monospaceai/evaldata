"""`MetricResultEquivalence`: decide equivalence by running both metric queries and diffing rows."""

from collections import Counter

from evaldata.dbt.errors import DbtError
from evaldata.dbt.metricflow import run
from evaldata.dbt.semantic_layer import MetricCase, MetricQuery
from evaldata.types import ScoreResult

SCORER_NAME = "metric_result_equivalence"


def _row_multiset(rows: list[dict[str, str]]) -> Counter[tuple[tuple[str, str], ...]]:
    return Counter(tuple(sorted(row.items())) for row in rows)


class MetricResultEquivalence:
    """Runs the candidate and gold queries through MetricFlow and compares their result rows.

    Equal result rows (as an order-insensitive multiset) pass; differing rows fail — both observed
    decisions. When either query cannot be run (MetricFlow unavailable, or the query does not run),
    the result is inconclusive.
    """

    def score(self, case: MetricCase, query: MetricQuery) -> ScoreResult:
        """Run both queries and decide equivalence from their result rows.

        Args:
            case: The eval case, supplying the gold query and the target directory.
            query: The candidate metric query.

        Returns:
            A passing or failing, observed `ScoreResult` when both queries run, else an
            inconclusive result.
        """
        candidate = run(query, case.target_dir)
        if isinstance(candidate, DbtError):
            return _inconclusive(f"model query: {candidate.message}")
        gold = run(case.gold, case.target_dir)
        if isinstance(gold, DbtError):
            return _inconclusive(f"gold query: {gold.message}")

        if _row_multiset(candidate) == _row_multiset(gold):
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


def _inconclusive(detail: str) -> ScoreResult:
    """Return an inconclusive `ScoreResult` carrying `detail`."""
    return ScoreResult(scorer=SCORER_NAME, verdict="inconclusive", explanation=detail)
