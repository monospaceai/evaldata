"""`ScoreContext`: per-case capabilities injected into `Scorer.score`."""

from dataclasses import dataclass

from dataeval.scorers.query import QueryRunner


@dataclass(frozen=True)
class ScoreContext:
    """Per-case capabilities injected into `Scorer.score`.

    A frozen struct carrying the handles a scorer needs while scoring one case.

    Attributes:
        queries: The budget-aware runner for derived SQL against the case's platform.
    """

    queries: QueryRunner
