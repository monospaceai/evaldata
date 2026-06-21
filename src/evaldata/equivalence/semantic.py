"""Pure combination of ordered `SemanticVerdict`s into a single `ScoreResult`."""

from evaldata.types import ScoreResult, SemanticVerdict


def combine(verdicts: list[SemanticVerdict], *, scorer: str) -> ScoreResult:
    """Combine ordered equivalence verdicts into one pass/fail `ScoreResult`.

    The first decisive verdict (not `"unknown"`) determines the outcome: `"equivalent"`
    passes, `"not_equivalent"` fails. When every verdict is `"unknown"` the result fails.
    Every verdict is recorded in `metadata["verdicts"]`; a refuting verdict's `diff` is
    surfaced on the result.

    Args:
        verdicts: The verdicts the checks produced, in the order they ran.
        scorer: The scorer name to stamp on the `ScoreResult`.

    Returns:
        A `ScoreResult` that passes iff the first decisive verdict is `"equivalent"`.
    """
    metadata = {"verdicts": [v.model_dump() for v in verdicts]}
    decisive = next((v for v in verdicts if v.equivalence != "unknown"), None)
    if decisive is None:
        return ScoreResult(
            scorer=scorer, passed=False, explanation="no check could decide equivalence", metadata=metadata
        )
    return ScoreResult(
        scorer=scorer, passed=decisive.equivalence == "equivalent", diff=decisive.diff, metadata=metadata
    )
