"""Tests for `PromptSolver` — the LLM-backed solver, driven through a `StubLlm`.

The solver talks only to the `Llm` seam, so its unit tests inject a `StubLlm` (no litellm,
no network). The litellm backend itself is covered by its own tests.
"""

import os

import pytest

from evaldata.llm import StubLlm, TextCompletion, Usage
from evaldata.solvers import SCHEMA_PROMPT_TEMPLATE, Solver
from evaldata.solvers.prompt import PromptSolver
from evaldata.types import (
    DuckDBPlatformRef,
    EvalCase,
    LlmError,
    ProviderErrorKind,
    SolverFailure,
    SolverSuccess,
    SQLDialect,
    UntypedResultSet,
)

_E2E_MODEL = "openai/gpt-4o-mini"


class _FixedLlm:
    """A minimal `Llm` returning a fixed text reply and `Usage`, to test telemetry passthrough."""

    def __init__(self, text: str, usage: Usage) -> None:
        self._text = text
        self._usage = usage

    def complete_text(self, prompt: str) -> TextCompletion:
        return TextCompletion(text=self._text, usage=self._usage)


def _case(*, dialect: SQLDialect | None = None) -> EvalCase:
    return EvalCase(
        id="c",
        input="How many tracks?",
        expected=UntypedResultSet(rows=[]),
        platform=DuckDBPlatformRef(name="local", dialect=dialect),
    )


@pytest.mark.unit
class TestPromptSolver:
    def test_happy_path(self) -> None:
        out = PromptSolver(model=StubLlm("SELECT 1 AS n")).solve(_case())
        assert isinstance(out, SolverSuccess)
        assert out.output == "SELECT 1 AS n"
        assert out.metadata["model"] == "StubLlm"

    def test_sql_is_stripped(self) -> None:
        out = PromptSolver(model=StubLlm("  SELECT 1  ")).solve(_case())
        assert isinstance(out, SolverSuccess)
        assert out.output == "SELECT 1"

    def test_code_fence_is_stripped(self) -> None:
        out = PromptSolver(model=StubLlm("```sql\nSELECT 1\n```")).solve(_case())
        assert isinstance(out, SolverSuccess)
        assert out.output == "SELECT 1"

    def test_empty_reply_is_empty_response(self) -> None:
        out = PromptSolver(model=StubLlm("")).solve(_case())
        assert isinstance(out, SolverFailure)
        assert out.error.kind == "empty_response"

    def test_whitespace_only_reply_is_empty_response(self) -> None:
        out = PromptSolver(model=StubLlm("   \n  ")).solve(_case())
        assert isinstance(out, SolverFailure)
        assert out.error.kind == "empty_response"

    def test_malformed_output_is_invalid_structured_output(self) -> None:
        err = LlmError(kind="malformed_output", message="model returned malformed structured output")
        out = PromptSolver(model=StubLlm(err)).solve(_case())
        assert isinstance(out, SolverFailure)
        assert out.error.kind == "invalid_structured_output"

    @pytest.mark.parametrize(
        "kind",
        [
            "timeout",
            "rate_limit",
            "auth",
            "context_window_exceeded",
            "bad_request",
            "api_connection",
            "api_error",
        ],
    )
    def test_provider_error_maps_one_to_one(self, kind: ProviderErrorKind) -> None:
        err = LlmError(kind=kind, message="provider failed", provider="openai")
        out = PromptSolver(model=StubLlm(err)).solve(_case())
        assert isinstance(out, SolverFailure)
        assert out.error.kind == kind
        assert out.error.provider == "openai"

    def test_telemetry_passthrough(self) -> None:
        usage = Usage(prompt_tokens=11, completion_tokens=7, cost_usd=0.0003, latency_seconds=1.5)
        out = PromptSolver(model=_FixedLlm("SELECT 1", usage)).solve(_case())
        assert out.prompt_tokens == 11
        assert out.completion_tokens == 7
        assert out.cost_usd == 0.0003
        assert out.latency_seconds == 1.5

    def test_dialect_injected_into_prompt(self) -> None:
        stub = StubLlm("SELECT 1")
        PromptSolver(model=stub).solve(_case(dialect="duckdb"))
        assert "duckdb" in stub.prompts[-1]

    def test_dialect_falls_back_to_kind(self) -> None:
        stub = StubLlm("SELECT 1")
        PromptSolver(model=stub).solve(_case(dialect=None))  # kind is "duckdb"
        assert "duckdb" in stub.prompts[-1]

    def test_custom_template_rendered(self) -> None:
        stub = StubLlm("SELECT 1")
        PromptSolver(model=stub, prompt_template="DIALECT={dialect} Q={input}").solve(_case(dialect="postgres"))
        assert stub.prompts[-1] == "DIALECT=postgres Q=How many tracks?"

    def test_schema_ddl_injected_with_schema_template(self) -> None:
        stub = StubLlm("SELECT 1")
        case = EvalCase(
            id="c",
            input="how many tracks?",
            expected=UntypedResultSet(rows=[]),
            platform=DuckDBPlatformRef(name="local"),
            metadata={"schema_ddl": "CREATE TABLE tracks (id INTEGER)"},
        )
        PromptSolver(model=stub, prompt_template=SCHEMA_PROMPT_TEMPLATE).solve(case)
        assert "CREATE TABLE tracks (id INTEGER)" in stub.prompts[-1]

    def test_satisfies_solver_protocol(self) -> None:
        assert isinstance(PromptSolver(model=StubLlm("SELECT 1")), Solver)


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="set OPENAI_API_KEY to run live solver e2e",
)
def test_live_prompt_solver_smoke() -> None:
    case = EvalCase(
        id="live",
        input="Return the single integer 1 as a column named n.",
        expected=UntypedResultSet(rows=[{"n": 1}]),
        platform=DuckDBPlatformRef(name="local", dialect="duckdb"),
    )
    out = PromptSolver(model=_E2E_MODEL).solve(case)
    assert isinstance(out, SolverSuccess)
    assert out.output.strip() != ""
