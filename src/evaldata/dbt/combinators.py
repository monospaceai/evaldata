"""`MetricFirstDecisive`: run member `MetricScorer`s in order, stopping at the first decisive result."""

from collections.abc import Sequence

from evaldata.dbt.semantic_layer import MetricCase, MetricQuery, MetricScorer
from evaldata.types import ScoreResult


class MetricFirstDecisive:
    """Runs member scorers in order; the first that decides wins, else the last result stands.

    Members run while each is `inconclusive`; the first decisive verdict (`pass` or `fail`) is
    returned immediately, so a later member cannot override an earlier decision. If every member is
    `inconclusive`, the last member's result is returned.
    """

    def __init__(self, scorers: Sequence[MetricScorer]) -> None:
        """Bind the combinator to an ordered list of member scorers.

        Args:
            scorers: The member scorers, in priority order.

        Raises:
            ValueError: If `scorers` is empty.
        """
        self._scorers = list(scorers)
        if not self._scorers:
            msg = "MetricFirstDecisive requires at least one scorer"
            raise ValueError(msg)

    def score(self, case: MetricCase, query: MetricQuery) -> ScoreResult:
        """Run members in order, returning the first decisive result, else the last.

        Args:
            case: The eval case, forwarded to each member.
            query: The candidate metric query, forwarded to each member.

        Returns:
            The first decisive member's `ScoreResult` (verdict `pass` or `fail`), or the last
            member's result when every member is `inconclusive`, with the `"first_decisive"` trail
            merged into its metadata.
        """
        trail: list[dict[str, object]] = []
        decided: ScoreResult | None = None
        for scorer in self._scorers:
            decided = scorer.score(case, query)
            trail.append({"scorer": decided.scorer, "passed": decided.passed, "verdict": decided.verdict})
            if decided.verdict != "inconclusive":
                break
        assert decided is not None
        return decided.model_copy(update={"metadata": {**decided.metadata, "first_decisive": trail}})
