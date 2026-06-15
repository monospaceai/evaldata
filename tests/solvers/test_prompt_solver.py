"""Tests for `PromptSolver` — the litellm-backed solver."""

import os
import types

import litellm
import pytest

from dataeval.solvers import Solver
from dataeval.solvers.prompt import PromptSolver, SqlOutput
from dataeval.types import EvalCase, PlatformRef, SQLDialect, UntypedResultSet

_E2E_MODEL = "openai/gpt-4o-mini"


def _case(*, dialect: SQLDialect | None = None) -> EvalCase:
    return EvalCase(
        id="c",
        input="How many tracks?",
        expected=UntypedResultSet(rows=[]),
        platform=PlatformRef(name="local", kind="duckdb", dialect=dialect),
    )


def _stub_response(content: str | None, *, prompt_tokens: int = 3, completion_tokens: int = 5, model: str = "gpt-stub"):
    """Build a SimpleNamespace exposing exactly what PromptSolver reads off a response."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return types.SimpleNamespace(choices=[choice], usage=usage, model=model)


def _patch_completion(
    monkeypatch: pytest.MonkeyPatch, response, captured: dict | None = None, *, structured: bool = False
) -> None:
    def fake(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return response

    monkeypatch.setattr("litellm.completion", fake)
    monkeypatch.setattr("litellm.supports_response_schema", lambda **kwargs: structured)


@pytest.mark.unit
class TestPromptSolver:
    def test_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("SELECT 1 AS n", model="gpt-4o-mini"))
        out = PromptSolver(model="gpt-4o-mini").solve(_case())
        assert out.error is None
        assert out.output == "SELECT 1 AS n"
        assert out.latency_seconds is not None
        assert out.latency_seconds >= 0
        assert out.metadata["model"] == "gpt-4o-mini"

    def test_fenced_output_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("```sql\nSELECT 1\n```"))
        out = PromptSolver(model="m").solve(_case())
        assert out.output == "SELECT 1"

    def test_prose_plus_fence(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("Here is the SQL:\n```sql\nSELECT 2\n```"))
        out = PromptSolver(model="m").solve(_case())
        assert out.output == "SELECT 2"

    def test_no_fence_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("SELECT id FROM t"))
        out = PromptSolver(model="m").solve(_case())
        assert out.output == "SELECT id FROM t"

    def test_empty_fence_falls_back_to_raw_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A fence with nothing inside it: the whole text is returned rather than an empty SQL.
        _patch_completion(monkeypatch, _stub_response("```sql\n```"))
        out = PromptSolver(model="m").solve(_case())
        assert out.output == "```sql\n```"

    def test_empty_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response(""))
        out = PromptSolver(model="m").solve(_case())
        assert out.output is None
        assert out.error is not None
        assert out.error.kind == "empty_response"

    def test_whitespace_only_is_empty_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("   \n  "))
        out = PromptSolver(model="m").solve(_case())
        assert out.error is not None
        assert out.error.kind == "empty_response"

    def test_none_content_is_empty_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response(None))
        out = PromptSolver(model="m").solve(_case())
        assert out.error is not None
        assert out.error.kind == "empty_response"

    def test_structured_output_returns_sql(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch_completion(monkeypatch, _stub_response('{"sql": "SELECT 1"}'), captured, structured=True)
        out = PromptSolver(model="gpt-4o-mini").solve(_case())
        assert out.error is None
        assert out.output == "SELECT 1"
        assert captured["response_format"] is SqlOutput

    def test_unsupported_model_falls_back_to_regex(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch_completion(monkeypatch, _stub_response("```sql\nSELECT 9\n```"), captured, structured=False)
        out = PromptSolver(model="m").solve(_case())
        assert out.output == "SELECT 9"
        assert "response_format" not in captured

    def test_structured_malformed_json_is_invalid_structured_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("SELECT 1 (not json)"), structured=True)
        out = PromptSolver(model="gpt-4o-mini").solve(_case())
        assert out.output is None
        assert out.error is not None
        assert out.error.kind == "invalid_structured_output"

    def test_structured_empty_sql_field_is_empty_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response('{"sql": ""}'), structured=True)
        out = PromptSolver(model="gpt-4o-mini").solve(_case())
        assert out.error is not None
        assert out.error.kind == "empty_response"

    def test_structured_none_content_is_invalid_structured_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # No reply content normalises to `{}`, which fails schema validation (sql is required).
        _patch_completion(monkeypatch, _stub_response(None), structured=True)
        out = PromptSolver(model="gpt-4o-mini").solve(_case())
        assert out.output is None
        assert out.error is not None
        assert out.error.kind == "invalid_structured_output"

    def test_structured_whitespace_content_is_invalid_structured_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("   \n "), structured=True)
        out = PromptSolver(model="gpt-4o-mini").solve(_case())
        assert out.error is not None
        assert out.error.kind == "invalid_structured_output"

    def test_token_and_cost_mapping(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("SELECT 1", prompt_tokens=11, completion_tokens=7))
        monkeypatch.setattr("litellm.completion_cost", lambda **kwargs: 0.0003)
        out = PromptSolver(model="m").solve(_case())
        assert out.prompt_tokens == 11
        assert out.completion_tokens == 7
        assert out.cost_usd == 0.0003

    def test_cost_unavailable_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_completion(monkeypatch, _stub_response("SELECT 1"))

        def boom(**kwargs):
            msg = "no pricing for this model"
            raise Exception(msg)

        monkeypatch.setattr("litellm.completion_cost", boom)
        out = PromptSolver(model="m").solve(_case())
        assert out.cost_usd is None
        assert out.output == "SELECT 1"

    def test_timeout_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):
            raise litellm.Timeout(message="timed out", model="m", llm_provider="openai")

        monkeypatch.setattr("litellm.completion", fake)
        out = PromptSolver(model="m").solve(_case())
        assert out.output is None
        assert out.error is not None
        assert out.error.kind == "timeout"
        assert out.error.provider == "openai"

    def test_rate_limit_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):
            raise litellm.RateLimitError(message="slow down", llm_provider="openai", model="m")

        monkeypatch.setattr("litellm.completion", fake)
        out = PromptSolver(model="m").solve(_case())
        assert out.output is None
        assert out.error is not None
        assert out.error.kind == "rate_limit"
        assert out.error.provider == "openai"

    def test_bad_request_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):
            raise litellm.BadRequestError(message="nope", model="m", llm_provider="openai")

        monkeypatch.setattr("litellm.completion", fake)
        out = PromptSolver(model="m").solve(_case())
        assert out.error is not None
        assert out.error.kind == "bad_request"

    def test_api_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):
            raise litellm.APIConnectionError(message="unreachable", llm_provider="openai", model="m")

        monkeypatch.setattr("litellm.completion", fake)
        out = PromptSolver(model="m").solve(_case())
        assert out.error is not None
        assert out.error.kind == "api_connection"

    def test_api_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):
            raise litellm.APIError(status_code=500, message="boom", llm_provider="openai", model="m")

        monkeypatch.setattr("litellm.completion", fake)
        out = PromptSolver(model="m").solve(_case())
        assert out.error is not None
        assert out.error.kind == "api_error"

    def test_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake(**kwargs):
            raise litellm.AuthenticationError(message="bad key", llm_provider="openai", model="m")

        monkeypatch.setattr("litellm.completion", fake)
        out = PromptSolver(model="m").solve(_case())
        assert out.output is None
        assert out.error is not None
        assert out.error.kind == "auth"

    def test_context_window_before_bad_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ContextWindowExceededError subclasses BadRequestError; it must map to its own kind.
        def fake(**kwargs):
            raise litellm.ContextWindowExceededError(message="too long", model="m", llm_provider="openai")

        monkeypatch.setattr("litellm.completion", fake)
        out = PromptSolver(model="m").solve(_case())
        assert out.error is not None
        assert out.error.kind == "context_window_exceeded"

    def test_dialect_injected_into_prompt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch_completion(monkeypatch, _stub_response("SELECT 1"), captured)
        PromptSolver(model="m").solve(_case(dialect="duckdb"))
        content = captured["messages"][0]["content"]
        assert "duckdb" in content

    def test_dialect_falls_back_to_kind(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch_completion(monkeypatch, _stub_response("SELECT 1"), captured)
        PromptSolver(model="m").solve(_case(dialect=None))  # kind is "duckdb"
        content = captured["messages"][0]["content"]
        assert "duckdb" in content

    def test_custom_template_rendered(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch_completion(monkeypatch, _stub_response("SELECT 1"), captured)
        PromptSolver(model="m", prompt_template="DIALECT={dialect} Q={input}").solve(_case(dialect="postgres"))
        content = captured["messages"][0]["content"]
        assert content == "DIALECT=postgres Q=How many tracks?"

    def test_timeout_passed_through(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict = {}
        _patch_completion(monkeypatch, _stub_response("SELECT 1"), captured)
        PromptSolver(model="m", timeout=12.5).solve(_case())
        assert captured["timeout"] == 12.5

    def test_satisfies_solver_protocol(self) -> None:
        assert isinstance(PromptSolver(model="gpt-4o-mini"), Solver)


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
        platform=PlatformRef(name="local", kind="duckdb", dialect="duckdb"),
    )
    out = PromptSolver(model=_E2E_MODEL).solve(case)
    assert out.error is None
    assert out.output is not None
    assert out.output.strip() != ""
