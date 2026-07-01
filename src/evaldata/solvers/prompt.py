"""`PromptSolver`: a single-prompt, LLM-backed `Solver` over the `Llm` seam."""

import re

from evaldata.llm import Llm, resolve_llm
from evaldata.solvers.errors import to_solver_error
from evaldata.types import EvalCase, LlmError, SolverError, SolverOutput, Sql

DEFAULT_PROMPT_TEMPLATE = """Generate a {dialect} SQL query that answers the following question.
Return only the SQL query with no explanation or markdown.

Question: {input}
SQL:
"""

SCHEMA_PROMPT_TEMPLATE = """Generate a {dialect} SQL query that answers the following question.
Use only the tables and columns in the schema below.
Return only the SQL query with no explanation or markdown.

Schema:
{schema}

Question: {input}
SQL:
"""

_SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)


def _extract_sql(text: str) -> str:
    """Pull the SQL out of a free-text reply, tolerating a Markdown code fence.

    Args:
        text: The raw model reply, possibly wrapped in a ```sql ... ``` fence.

    Returns:
        The fenced block's contents when present, else the stripped text.
    """
    fence = _SQL_FENCE_RE.search(text)
    if fence is not None:
        return fence.group(1).strip()
    return text.strip()


class PromptSolver:
    """Single-prompt LLM `Solver`: question -> SQL via the `Llm` seam."""

    def __init__(
        self,
        model: str | Llm,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:
        """Configure the solver.

        Args:
            model: A litellm model identifier (e.g. `"openai/gpt-4o-mini"`), or an `Llm` to use
                directly. `timeout` and `temperature` apply only to the model-string path.
            prompt_template: A `str.format_map` template with `{dialect}`, `{input}`, and
                optional `{schema}` fields. Defaults to `DEFAULT_PROMPT_TEMPLATE`; pass
                `SCHEMA_PROMPT_TEMPLATE` to inject `case.metadata["schema_ddl"]`.
            timeout: Per-request timeout in seconds.
            temperature: Sampling temperature; `None` leaves the provider default. Use `0` for
                deterministic output.
        """
        self._llm = resolve_llm(model, temperature=temperature, timeout=timeout)
        self._model = model if isinstance(model, str) else type(model).__name__
        self._prompt_template = prompt_template

    def solve(self, case: EvalCase) -> SolverOutput:
        """Produce SQL for `case`, returning a success or a typed `SolverError`.

        Renders the prompt, calls the model, and extracts the SQL. Expected provider failures
        are mapped to a `SolverError` in `SolverOutput.error`.

        Args:
            case: The eval case to solve.

        Returns:
            A `SolverOutput` carrying either the extracted SQL plus token/latency/cost
            telemetry on success, or a typed `SolverError` on an expected failure.
        """
        dialect = case.platform.dialect or case.platform.kind
        schema = case.metadata.get("schema_ddl", "")
        prompt = self._prompt_template.format_map({"dialect": dialect, "input": case.input, "schema": schema})
        completion = self._llm.complete_text(prompt)
        if isinstance(completion, LlmError):
            return SolverOutput(error=to_solver_error(completion))

        sql = Sql(_extract_sql(completion.text).strip())
        if not sql:
            return SolverOutput(
                error=SolverError(kind="empty_response", message="model returned no SQL", provider=None)
            )
        usage = completion.usage
        return SolverOutput(
            output=sql,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_seconds=usage.latency_seconds,
            cost_usd=usage.cost_usd,
            metadata={"model": self._model},
        )
