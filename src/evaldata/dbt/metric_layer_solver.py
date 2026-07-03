"""`MetricLayerSolver`: an LLM `MetricSolver` that answers a question with a dbt Semantic Layer query."""

from evaldata.dbt.semantic_layer import MetricCase, MetricQuery, MetricSolverOutput
from evaldata.llm import Llm, resolve_llm
from evaldata.solvers.errors import to_solver_error
from evaldata.types import LlmError

SL_PROMPT_TEMPLATE = """You are querying a dbt Semantic Layer with MetricFlow. Answer the question with
metrics and group-by items chosen only from the semantic layer below. Every group-by item must be
one of the exact names shown below: a dimension is referenced through its entity
with a double underscore (for example `customer__country`), and a time dimension takes a grain by
appending it (for example `metric_time__month`).

Semantic layer:
{semantic_layer}

Question: {input}
"""


class MetricLayerSolver:
    """Single-prompt LLM `MetricSolver`: question -> a `MetricQuery` via structured output."""

    def __init__(
        self,
        model: str | Llm,
        prompt_template: str = SL_PROMPT_TEMPLATE,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:
        """Configure the solver.

        Args:
            model: A litellm model identifier (e.g. `"openai/gpt-4o-mini"`), or an `Llm` to use
                directly. `timeout` and `temperature` apply only to the model-string path.
            prompt_template: A `str.format_map` template with `{semantic_layer}` and `{input}`
                fields; `{semantic_layer}` is filled from `case.sl_context`.
            timeout: Per-request timeout in seconds.
            temperature: Sampling temperature; `None` leaves the provider default. Use `0` for
                deterministic output.
        """
        self._llm = resolve_llm(model, temperature=temperature, timeout=timeout)
        self._model = model if isinstance(model, str) else type(model).__name__
        self._prompt_template = prompt_template

    def solve(self, case: MetricCase) -> MetricSolverOutput:
        """Produce a metric query for `case`.

        Args:
            case: The eval case to solve.

        Returns:
            A `MetricSolverOutput` with the metric query and telemetry on success, or a typed
            `SolverError` on an expected provider failure.
        """
        prompt = self._prompt_template.format_map({"input": case.input, "semantic_layer": case.sl_context})
        completion = self._llm.complete(prompt, response_format=MetricQuery)
        if isinstance(completion, LlmError):
            return MetricSolverOutput(error=to_solver_error(completion))

        usage = completion.usage
        return MetricSolverOutput(
            query=completion.parsed,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_seconds=usage.latency_seconds,
            cost_usd=usage.cost_usd,
            metadata={"model": self._model},
        )
