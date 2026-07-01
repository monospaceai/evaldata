"""`MetricSpecEquivalence`: confirm two metric queries match by resolving both through MetricFlow."""

from evaldata.dbt.errors import DbtError
from evaldata.dbt.metricflow import canonicalize
from evaldata.dbt.semantic_layer import MetricCase, MetricQuery
from evaldata.types import ScoreResult

SCORER_NAME = "metric_spec_equivalence"


class MetricSpecEquivalence:
    """Compares two metric queries by the forms MetricFlow resolves them to.

    Equal forms pass (proven). A candidate MetricFlow rejects fails (proven). Everything else —
    unequal forms, an invalid gold, or the toolchain being unavailable — is inconclusive.
    """

    def score(self, case: MetricCase, query: MetricQuery) -> ScoreResult:
        """Resolve the candidate and gold queries and confirm equivalence when they match.

        Args:
            case: The eval case, supplying the gold query and the target directory.
            query: The candidate metric query.

        Returns:
            A proven pass when the queries resolve to the same form, a proven fail when the
            candidate does not resolve against the manifest, else an inconclusive result.
        """
        candidate = canonicalize(query, case.target_dir)
        if isinstance(candidate, DbtError):
            if candidate.kind == "metric_query_invalid":
                return ScoreResult(
                    scorer=SCORER_NAME, verdict="fail", basis="proven", explanation=f"model query: {candidate.message}"
                )
            return _inconclusive(f"model query: {candidate.message}")
        gold = canonicalize(case.gold, case.target_dir)
        if isinstance(gold, DbtError):
            return _inconclusive(f"gold query: {gold.message}")

        if candidate == gold:
            return ScoreResult(
                scorer=SCORER_NAME,
                verdict="pass",
                basis="proven",
                explanation="metric queries resolve to the same MetricFlow query",
            )
        return _inconclusive("metric queries resolve to different MetricFlow queries")


def _inconclusive(detail: str) -> ScoreResult:
    return ScoreResult(scorer=SCORER_NAME, verdict="inconclusive", explanation=detail)
