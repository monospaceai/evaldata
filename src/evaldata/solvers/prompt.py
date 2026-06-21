"""`PromptSolver`: a single-prompt, LLM-backed `Solver` over `litellm`."""

import re
import time

import litellm
from pydantic import BaseModel, ValidationError

from evaldata.types import EvalCase, SolverError, SolverErrorKind, SolverOutput, Sql

DEFAULT_PROMPT_TEMPLATE = """Generate a {dialect} SQL query that answers the following question.
Return only the SQL query with no explanation or markdown.

Question: {input}
SQL:
"""

_FENCE_RE = re.compile(r"```(?:sql)?\s*([\s\S]*?)```", re.IGNORECASE)


class SqlOutput(BaseModel):
    """Structured solver reply: the generated SQL query."""

    sql: str


def _extract_structured_sql(content: str | None) -> Sql | SolverError:
    """Parse a structured `{sql: ...}` reply, returning the SQL or a typed error.

    Empty or absent content normalises to `{}`, so a missing reply fails validation like
    malformed JSON.

    Args:
        content: The raw model reply, expected to be a JSON object matching `SqlOutput`.

    Returns:
        The stripped SQL (possibly empty when the `sql` field is empty), or a `SolverError`
        of kind `invalid_structured_output` when the content does not validate against
        `SqlOutput`.
    """
    try:
        reply = SqlOutput.model_validate_json(content or "{}")
    except ValidationError as e:
        return SolverError(
            kind="invalid_structured_output",
            message=f"model returned malformed structured output: {(content or '')[:200]!r}",
            provider=None,
            cause=e,
        )
    return Sql(reply.sql.strip())


def _extract_sql(text: str) -> str:
    """Extract SQL from raw model text, stripping a Markdown code fence if present.

    Args:
        text: The raw model reply, optionally wrapping the SQL in a fenced code block.

    Returns:
        The contents of the first fenced code block when non-empty, otherwise the whole
        text stripped of surrounding whitespace.
    """
    match = _FENCE_RE.search(text)
    if match is not None:
        inner = match.group(1).strip()
        if inner:
            return inner
    return text.strip()


class PromptSolver:
    """Single-prompt LLM `Solver`: question -> SQL via `litellm.completion`."""

    def __init__(
        self,
        model: str,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
        timeout: float | None = None,
        temperature: float | None = None,
    ) -> None:
        """Configure the solver.

        Args:
            model: The litellm model identifier (e.g. `"openai/gpt-4o-mini"`). Required.
            prompt_template: A `str.format_map` template with `{dialect}` and
                `{input}` fields. Defaults to `DEFAULT_PROMPT_TEMPLATE`.
            timeout: Per-request timeout in seconds.
            temperature: Sampling temperature; `None` leaves the provider default.
                Use `0` for deterministic output.
        """
        self._model = model
        self._prompt_template = prompt_template
        self._timeout = timeout
        self._temperature = temperature

    def solve(self, case: EvalCase) -> SolverOutput:
        """Produce SQL for `case`, returning a success or a typed `SolverError`.

        Renders the prompt, calls the model, and extracts the SQL. Expected provider failures
        are mapped to a `SolverError` and returned as `SolverOutput.error`, not raised.

        Args:
            case: The eval case to solve.

        Returns:
            A `SolverOutput` carrying either the extracted SQL plus token/latency/cost
            telemetry on success, or a typed `SolverError` on an expected failure.
        """
        dialect = case.platform.dialect or case.platform.kind
        rendered = self._prompt_template.format_map({"dialect": dialect, "input": case.input})
        messages = [{"role": "user", "content": rendered}]
        structured = litellm.supports_response_schema(model=self._model)
        kwargs: dict = {"model": self._model, "messages": messages, "timeout": self._timeout}
        if structured:
            kwargs["response_format"] = SqlOutput
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        start = time.perf_counter()
        try:
            response = litellm.completion(**kwargs)
        except litellm.Timeout as e:
            return SolverOutput(error=self._error("timeout", e))
        except litellm.RateLimitError as e:
            return SolverOutput(error=self._error("rate_limit", e))
        except litellm.AuthenticationError as e:
            return SolverOutput(error=self._error("auth", e))
        except litellm.ContextWindowExceededError as e:
            return SolverOutput(error=self._error("context_window_exceeded", e))
        except litellm.BadRequestError as e:
            return SolverOutput(error=self._error("bad_request", e))
        except litellm.APIConnectionError as e:
            return SolverOutput(error=self._error("api_connection", e))
        except litellm.APIError as e:
            return SolverOutput(error=self._error("api_error", e))
        elapsed = time.perf_counter() - start

        content = response.choices[0].message.content
        if structured:
            extracted = _extract_structured_sql(content)
            if isinstance(extracted, SolverError):
                return SolverOutput(error=extracted)
            sql = extracted
        else:
            sql = _extract_sql(content) if content is not None else ""
        if not sql:
            return SolverOutput(
                error=SolverError(kind="empty_response", message="model returned no SQL", provider=None)
            )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        try:
            cost = litellm.completion_cost(completion_response=response)
        except Exception:
            # Local/unknown models have no pricing table; cost is simply unavailable.
            cost = None

        return SolverOutput(
            output=Sql(sql),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_seconds=elapsed,
            cost_usd=cost,
            metadata={"model": getattr(response, "model", self._model)},
        )

    @staticmethod
    def _error(kind: SolverErrorKind, exc: Exception) -> SolverError:
        """Build a `SolverError` from a litellm exception, capturing `llm_provider`.

        Args:
            kind: The typed error category.
            exc: The litellm exception to wrap.

        Returns:
            A `SolverError` carrying the kind, message, and provider (if available).
        """
        return SolverError(
            kind=kind, message=str(exc) or type(exc).__name__, provider=getattr(exc, "llm_provider", None), cause=exc
        )
