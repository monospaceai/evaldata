"""`LlmJudge`: a probabilistic LLM-as-judge `Scorer` over the `Llm` seam.

A grader model scores the case against authored criteria; its 0-1 score maps to a pass/fail
verdict, or an inconclusive result when no verdict can be reached.
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, model_validator

from evaldata.llm import Llm, resolve_llm
from evaldata.scorers.context import ScoreContext
from evaldata.types import EvalCase, ExecutionResult, GoldQuery, LlmError, Score, ScoreResult, SolverOutput

SCORER_NAME = "llm_judge"

# A case field shown to the grader: the input, the actual output, or the expected output.
JudgeField = Literal["input", "actual_output", "expected_output"]

_ALL_FIELDS: tuple[JudgeField, ...] = ("input", "actual_output", "expected_output")

JUDGE_INSTRUCTION = (
    "You are an expert SQL reviewer. Grade the actual output below against the criteria. "
    "Reason about how well it meets each criterion, then assign a score from 0.0 (does not meet "
    "the criteria) to 1.0 (fully meets them)."
)

_OUTPUT_FORMAT = "Return a JSON object with a numeric `score` in [0.0, 1.0] and a short `reason`."


class JudgeReply(BaseModel):
    """Structured grader reply: the grader's rationale and a 0-1 correctness score."""

    # `reason` precedes `score` so the grader reasons before committing to a number.
    reason: str
    score: float


class JudgeExample(BaseModel):
    """A few-shot anchor: a graded output and the score it should receive.

    `input` and `expected_output` are optional context; only the fields that are present are
    rendered into the prompt.
    """

    actual_output: str
    score: Score
    reason: str
    input: str | None = None
    expected_output: str | None = None


class RubricBand(BaseModel):
    """A scoring band: a `[min_score, max_score]` range and what it describes."""

    min_score: Score
    max_score: Score
    description: str

    @model_validator(mode="after")
    def _ordered_bounds(self) -> "RubricBand":
        """Reject a band whose `min_score` exceeds its `max_score`.

        Returns:
            The validated `RubricBand`.

        Raises:
            ValueError: If `min_score` is greater than `max_score`.
        """
        if self.min_score > self.max_score:
            msg = "RubricBand min_score cannot exceed max_score"
            raise ValueError(msg)
        return self


class LlmJudge:
    """LLM-as-judge `Scorer`: a grader model scores the case against authored criteria.

    The grader's 0-1 score is compared to a threshold for the pass/fail verdict; the score and
    rationale are recorded. A provider failure or a malformed reply yields an inconclusive
    result.
    """

    def __init__(
        self,
        *,
        model: str | Llm,
        criteria: str,
        steps: Sequence[str] | None = None,
        examples: Sequence[JudgeExample] | None = None,
        rubric: Sequence[RubricBand] | None = None,
        threshold: float = 0.5,
        temperature: float | None = 0.0,
        timeout: float | None = None,
        show: Sequence[JudgeField] | None = None,
    ) -> None:
        """Configure the judge.

        Args:
            model: A litellm grader-model identifier (separate from any solver model), or an
                `Llm` to use directly. `temperature` and `timeout` apply only to the
                model-string path.
            criteria: The natural-language standard the grader scores the case against.
            steps: Ordered evaluation steps the grader should work through, rendered as a
                numbered block. Omitted from the prompt when absent.
            examples: Few-shot anchors mapping graded outputs to scores. Omitted from the
                prompt when absent.
            rubric: Scoring bands that describe what each score range means. Omitted from the
                prompt when absent.
            threshold: The minimum score (inclusive) for a passing verdict. Defaults to `0.5`.
            temperature: Sampling temperature; `None` leaves the provider default. Defaults to
                `0.0` for deterministic grading.
            timeout: Per-request timeout in seconds.
            show: The case fields to offer the grader, each included only when available.
                Defaults to all of `input`, `actual_output`, and `expected_output`.
        """
        self._llm = resolve_llm(model, temperature=temperature, timeout=timeout)
        self._model = model if isinstance(model, str) else type(model).__name__
        self._criteria = criteria
        self._steps = tuple(steps) if steps is not None else ()
        self._examples = tuple(examples) if examples is not None else ()
        self._rubric = tuple(rubric) if rubric is not None else ()
        self._threshold = threshold
        self._show = tuple(show) if show is not None else _ALL_FIELDS

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Grade `case` with the grader model and return a graded `ScoreResult`.

        Builds a prompt from the criteria and the selected available fields, calls the grader,
        and maps its score to a verdict against the threshold.

        Args:
            case: The eval case, supplying the input and (optionally) the expected output.
            output: The solver output (part of the `Scorer` protocol; unused here).
            result: The executed model result (part of the `Scorer` protocol; unused here).
            context: The score context, supplying the model's SQL.

        Returns:
            A `ScoreResult` whose verdict is pass or fail with the graded score and rationale,
            or inconclusive when no verdict could be reached.
        """
        prompt = self._build_prompt(case, context)
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
        verdict = "pass" if clamped >= self._threshold else "fail"
        return ScoreResult(
            scorer=SCORER_NAME,
            verdict=verdict,
            score=clamped,
            basis="judged",
            explanation=completion.parsed.reason or None,
            metadata=metadata,
        )

    def _build_prompt(self, case: EvalCase, context: ScoreContext) -> str:
        """Render the grader prompt from the configured guidance and the available fields.

        Args:
            case: The eval case, supplying the input and (optionally) the expected output.
            context: The score context, supplying the model's SQL.

        Returns:
            The grader prompt: the framing, the criteria, any steps/rubric/examples,
            each available selected field, and the fixed JSON output-format request.
        """
        parts = [JUDGE_INSTRUCTION, f"Criteria:\n{self._criteria}"]

        if self._steps:
            numbered = "\n".join(f"{i}. {step}" for i, step in enumerate(self._steps, start=1))
            parts.append(f"Evaluation steps:\n{numbered}")

        if self._rubric:
            bands = "\n".join(f"- {band.min_score}-{band.max_score}: {band.description}" for band in self._rubric)
            parts.append(f"Scoring rubric:\n{bands}")

        if self._examples:
            parts.append(f"Examples:\n\n{self._render_examples()}")

        if "input" in self._show:
            parts.append(f"Input:\n{case.input}")
        if "actual_output" in self._show:
            parts.append(f"Actual Output:\n{context.queries.model_sql}")
        if "expected_output" in self._show and isinstance(case.expected, GoldQuery):
            parts.append(f"Expected Output:\n{case.expected.sql}")

        parts.append(_OUTPUT_FORMAT)
        return "\n\n".join(parts)

    def _render_examples(self) -> str:
        """Render the configured few-shot examples, each showing only its present fields.

        Returns:
            The examples joined by blank lines; each example lists its present context fields
            followed by its `Score:` and `Reason:`.
        """
        blocks = []
        for example in self._examples:
            lines = []
            if example.input is not None:
                lines.append(f"Input:\n{example.input}")
            lines.append(f"Actual Output:\n{example.actual_output}")
            if example.expected_output is not None:
                lines.append(f"Expected Output:\n{example.expected_output}")
            lines.append(f"Score: {example.score}")
            lines.append(f"Reason: {example.reason}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
