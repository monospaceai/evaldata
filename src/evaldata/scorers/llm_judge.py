"""`LlmJudge`: a probabilistic LLM-as-judge `Scorer` over `litellm`.

A grader model scores the case against authored criteria; its 0-1 score maps to a pass/fail
verdict, or an inconclusive result when no verdict can be reached.
"""

from collections.abc import Sequence
from typing import Literal

import litellm
from pydantic import BaseModel, ValidationError

from evaldata.scorers.context import ScoreContext
from evaldata.types import EvalCase, ExecutionResult, GoldQuery, ScoreResult, SolverOutput

SCORER_NAME = "llm_judge"

# A case field shown to the grader: the question, the candidate SQL, or the gold query.
JudgeField = Literal["question", "model_sql", "gold_query"]

_ALL_FIELDS: tuple[JudgeField, ...] = ("question", "model_sql", "gold_query")

_INSTRUCTION = (
    "You are grading a candidate SQL query against the criteria below. Judge how well it meets "
    "them and return JSON with a `score` between 0.0 (fails the criteria) and 1.0 (fully meets "
    "them) and a short `reason` explaining the score."
)


class JudgeReply(BaseModel):
    """Structured grader reply: a 0-1 correctness score and its rationale."""

    score: float
    reason: str


class LlmJudge:
    """LLM-as-judge `Scorer`: a grader model scores the case against authored criteria.

    The grader's 0-1 score is compared to a threshold for the pass/fail verdict; the score and
    rationale are recorded. A grader without structured-output support, a provider failure, or
    a malformed reply each yield an inconclusive result.
    """

    def __init__(
        self,
        *,
        model: str,
        criteria: str,
        threshold: float = 0.5,
        temperature: float | None = 0.0,
        timeout: float | None = None,
        show: Sequence[JudgeField] | None = None,
    ) -> None:
        """Configure the judge.

        Args:
            model: The litellm grader-model identifier (separate from any solver model).
            criteria: The natural-language standard the grader scores the case against.
            threshold: The minimum score (inclusive) for a passing verdict. Defaults to `0.5`.
            temperature: Sampling temperature; `None` leaves the provider default. Defaults to
                `0.0` for deterministic grading.
            timeout: Per-request timeout in seconds.
            show: The case fields to offer the grader, each included only when available.
                Defaults to all of `question`, `model_sql`, and `gold_query`.
        """
        self._model = model
        self._criteria = criteria
        self._threshold = threshold
        self._temperature = temperature
        self._timeout = timeout
        self._show = tuple(show) if show is not None else _ALL_FIELDS

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Grade `case` with the grader model and return a graded `ScoreResult`.

        Builds a prompt from the criteria and the selected available fields, calls the grader,
        and maps its score to a verdict against the threshold.

        Args:
            case: The eval case, supplying the question and (optionally) the gold query.
            output: The solver output (part of the `Scorer` protocol; unused here).
            result: The executed model result (part of the `Scorer` protocol; unused here).
            context: The score context, supplying the model's SQL.

        Returns:
            A `ScoreResult` whose verdict is pass or fail with the graded score and rationale,
            or inconclusive when no verdict could be reached.
        """
        prompt = self._build_prompt(case, context)
        metadata: dict = {"source": "llm_judge", "grader_model": self._model}

        if not litellm.supports_response_schema(model=self._model):
            return ScoreResult(
                scorer=SCORER_NAME,
                verdict="inconclusive",
                explanation=f"grader model {self._model} does not support structured output",
                metadata=metadata,
            )

        kwargs: dict = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": JudgeReply,
            "timeout": self._timeout,
        }
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature

        try:
            response = litellm.completion(**kwargs)
        except litellm.Timeout as e:
            return self._inconclusive("timeout", e, metadata)
        except litellm.RateLimitError as e:
            return self._inconclusive("rate_limit", e, metadata)
        except litellm.AuthenticationError as e:
            return self._inconclusive("auth", e, metadata)
        except litellm.ContextWindowExceededError as e:
            return self._inconclusive("context_window_exceeded", e, metadata)
        except litellm.BadRequestError as e:
            return self._inconclusive("bad_request", e, metadata)
        except litellm.APIConnectionError as e:
            return self._inconclusive("api_connection", e, metadata)
        except litellm.APIError as e:
            return self._inconclusive("api_error", e, metadata)

        content = response.choices[0].message.content
        try:
            reply = JudgeReply.model_validate_json(content or "{}")
        except ValidationError:
            return ScoreResult(
                scorer=SCORER_NAME,
                verdict="inconclusive",
                explanation="grader returned malformed output",
                metadata=metadata,
            )

        clamped = min(1.0, max(0.0, reply.score))
        verdict = "pass" if clamped >= self._threshold else "fail"
        return ScoreResult(
            scorer=SCORER_NAME,
            verdict=verdict,
            score=clamped,
            explanation=reply.reason or None,
            metadata=metadata,
        )

    def _build_prompt(self, case: EvalCase, context: ScoreContext) -> str:
        """Render the grader prompt from the criteria and the selected available fields.

        Args:
            case: The eval case, supplying the question and (optionally) the gold query.
            context: The score context, supplying the model's SQL.

        Returns:
            The grader prompt: the instruction, the criteria, each available selected field,
            and the JSON-output request.
        """
        parts = [_INSTRUCTION, f"Criteria:\n{self._criteria}"]
        if "question" in self._show:
            parts.append(f"Question:\n{case.input}")
        if "model_sql" in self._show:
            parts.append(f"Candidate SQL:\n{context.queries.model_sql}")
        if "gold_query" in self._show and isinstance(case.expected, GoldQuery):
            parts.append(f"Reference SQL:\n{case.expected.sql}")
        parts.append("Return JSON with a `score` between 0.0 and 1.0 and a `reason`.")
        return "\n\n".join(parts)

    @staticmethod
    def _inconclusive(kind: str, exc: Exception, metadata: dict) -> ScoreResult:
        """Build an inconclusive `ScoreResult` from a litellm exception.

        Args:
            kind: The short error category.
            exc: The litellm exception to surface.
            metadata: The base metadata to extend with the structured error.

        Returns:
            An inconclusive `ScoreResult` carrying the error in `explanation` and `metadata`.
        """
        message = str(exc) or type(exc).__name__
        return ScoreResult(
            scorer=SCORER_NAME,
            verdict="inconclusive",
            explanation=f"grader call failed: {message}",
            metadata={**metadata, "error": {"kind": kind, "message": message}},
        )
