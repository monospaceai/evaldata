"""`MetricLayerJudge`: an LLM-as-judge `MetricScorer` grading two metric queries for equivalence."""

from evaldata.dbt.semantic_layer import MetricCase, MetricQuery
from evaldata.llm import Llm, resolve_llm
from evaldata.scorers.llm_judge import JudgeReply
from evaldata.types import LlmError, ScoreResult

SCORER_NAME = "metric_layer_judge"

SL_JUDGE_CRITERIA = (
    "You review dbt Semantic Layer (MetricFlow) queries. Decide whether the candidate query is "
    "equivalent to the reference query: the same metrics, group-by items, filters, ordering, and "
    "limit. Ignore differences that never change the result, such as a default versus an explicit "
    "time grain (metric_time versus metric_time__day) or the ordering of group-by items. Reason "
    "briefly about each part, then score from 0.0 (different) to 1.0 (equivalent)."
)

_OUTPUT_FORMAT = "Return a JSON object with a numeric `score` in [0.0, 1.0] and a short `reason`."


def _render_query(query: MetricQuery) -> str:
    parts = [f"metrics: {', '.join(query.metrics)}"]
    if query.group_by:
        parts.append(f"group by: {', '.join(query.group_by)}")
    if query.where:
        parts.append(f"where: {' AND '.join(query.where)}")
    if query.order_by:
        parts.append(f"order by: {', '.join(query.order_by)}")
    if query.limit is not None:
        parts.append(f"limit: {query.limit}")
    return "\n".join(parts)


class MetricLayerJudge:
    """LLM-as-judge `MetricScorer`: a grader model scores the candidate query against the gold query.

    The grader's 0-1 score is compared to a threshold for the pass/fail verdict; the score and
    rationale are recorded. A provider failure or a malformed reply yields an inconclusive result.
    """

    def __init__(
        self,
        model: str | Llm,
        *,
        criteria: str = SL_JUDGE_CRITERIA,
        threshold: float = 0.5,
        temperature: float | None = 0.0,
        timeout: float | None = None,
    ) -> None:
        """Configure the judge.

        Args:
            model: A litellm grader-model identifier (separate from any solver model), or an `Llm`
                to use directly. `temperature` and `timeout` apply only to the model-string path.
            criteria: The natural-language standard the grader scores the queries against.
            threshold: The minimum score (inclusive) for a passing verdict.
            temperature: Sampling temperature; defaults to `0.0` for deterministic grading.
            timeout: Per-request timeout in seconds.
        """
        self._llm = resolve_llm(model, temperature=temperature, timeout=timeout)
        self._model = model if isinstance(model, str) else type(model).__name__
        self._criteria = criteria
        self._threshold = threshold

    def score(self, case: MetricCase, query: MetricQuery) -> ScoreResult:
        """Grade the candidate query against the gold query and return a graded `ScoreResult`.

        Args:
            case: The eval case, supplying the question and the gold query.
            query: The candidate metric query.

        Returns:
            A `ScoreResult` whose verdict is pass or fail with the graded score and rationale, or
            inconclusive when the grader call fails.
        """
        prompt = "\n\n".join(
            [
                self._criteria,
                f"Question:\n{case.input}",
                f"Candidate query:\n{_render_query(query)}",
                f"Reference query:\n{_render_query(case.gold)}",
                _OUTPUT_FORMAT,
            ]
        )
        metadata: dict = {"source": "llm_judge", "grader_model": self._model}
        completion = self._llm.complete(prompt, response_format=JudgeReply)
        if isinstance(completion, LlmError):
            return ScoreResult(
                scorer=SCORER_NAME,
                verdict="inconclusive",
                explanation=f"grader call failed: {completion.message}",
                metadata={**metadata, "error": {"kind": completion.kind, "message": completion.message}},
            )
        clamped = min(1.0, max(0.0, completion.parsed.score))
        return ScoreResult(
            scorer=SCORER_NAME,
            verdict="pass" if clamped >= self._threshold else "fail",
            score=clamped,
            basis="judged",
            explanation=completion.parsed.reason or None,
            metadata=metadata,
        )
