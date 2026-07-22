"""`FirstDecisive`: a generic scorer combinator running members until one decides."""

from collections.abc import Sequence

from evaldata.scorers.base import Scorer
from evaldata.scorers.context import ScoreContext
from evaldata.types import EvalCase, ExecutionResult, ScoreResult, SolverSuccess


class FirstDecisive:
    """Runs member scorers in order; the first that decides wins, else the last result stands.

    The combinator continues only while a member is `inconclusive`; the first member to return a
    decisive verdict (`pass` or `fail`) is returned immediately, so a later member cannot override
    an earlier one's decision. If every member is `inconclusive`, the last member's result is
    returned, so its diagnostics (e.g. a diff) surface.
    """

    def __init__(self, scorers: Sequence[Scorer]) -> None:
        """Bind the combinator to an ordered list of member scorers.

        Args:
            scorers: The member scorers, in priority order.

        Raises:
            ValueError: If `scorers` is empty.
        """
        self._scorers = list(scorers)
        if not self._scorers:
            msg = "FirstDecisive requires at least one scorer"
            raise ValueError(msg)

    def score(
        self, case: EvalCase, output: SolverSuccess, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Run members in order, returning the first decisive result (later members not consulted), else the last.

        The returned result carries a `metadata["first_decisive"]` trail of
        `{"scorer", "passed", "verdict"}` for each member that actually ran.

        Args:
            case: The eval case, forwarded to each member.
            output: The solver output, forwarded to each member.
            result: The executed model result, forwarded to each member.
            context: The score context, forwarded to each member.

        Returns:
            The first decisive member's `ScoreResult` (verdict `pass` or `fail`), or the last
            member's result when every member is `inconclusive`, with the `"first_decisive"` trail
            merged into its metadata.
        """
        first, *remaining = self._scorers
        decided = first.score(case, output, result, context=context)
        trail: list[dict[str, object]] = [
            {"scorer": decided.scorer, "passed": decided.passed, "verdict": decided.verdict}
        ]
        for scorer in remaining:
            if decided.verdict != "inconclusive":
                break
            decided = scorer.score(case, output, result, context=context)
            trail.append({"scorer": decided.scorer, "passed": decided.passed, "verdict": decided.verdict})
        return decided.model_copy(update={"metadata": {**decided.metadata, "first_decisive": trail}})
