"""Tests for `PromptSolver` — the litellm-backed solver."""

import os
import types

import litellm
import pytest

from data_eval.solvers import Solver
from data_eval.solvers.prompt import PromptSolver
from data_eval.types import EvalCase, ExpectedResultSet, PlatformRef, SQLDialect

_E2E_MODEL = "openai/gpt-4o-mini"


def _case(*, dialect: SQLDialect | None = None) -> EvalCase:
    return EvalCase(
        id="c",
        input="How many tracks?",
        expected=ExpectedResultSet(rows=[]),
        platform=PlatformRef(name="local", kind="duckdb", dialect=dialect),
    )


def _stub_response(content: str | None, *, prompt_tokens: int = 3, completion_tokens: int = 5, model: str = "gpt-stub"):
    """Build a SimpleNamespace exposing exactly what PromptSolver reads off a response."""
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    usage = types.SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return types.SimpleNamespace(choices=[choice], usage=usage, model=model)


def _patch_completion(monkeypatch: pytest.MonkeyPatch, response, captured: dict | None = None) -> None:
    def fake(**kwargs):
        if captured is not None:
            captured.update(kwargs)
        return response

    monkeypatch.setattr("litellm.completion", fake)


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
        expected=ExpectedResultSet(rows=[{"n": 1}]),
        platform=PlatformRef(name="local", kind="duckdb", dialect="duckdb"),
    )
    out = PromptSolver(model=_E2E_MODEL).solve(case)
    assert out.error is None
    assert out.output is not None
    assert out.output.strip() != ""
