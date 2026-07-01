"""Tests for `MetricLayerSolver`, driven through a `StubLlm` (no litellm, no network)."""

import pytest
from pydantic import ValidationError

from evaldata.dbt import MetricCase, MetricLayerSolver, MetricQuery, MetricSolver, MetricSolverOutput
from evaldata.llm import Completion, StubLlm, Usage
from evaldata.types import LlmError, PlatformRef, ProviderErrorKind, SolverError

pytestmark = pytest.mark.unit


class _FixedLlm:
    """A minimal `Llm` returning a fixed parsed reply and `Usage`, to test telemetry passthrough."""

    def __init__(self, query: MetricQuery, usage: Usage) -> None:
        self._query = query
        self._usage = usage

    def complete(self, prompt: str, *, response_format: type[MetricQuery]) -> Completion[MetricQuery]:
        return Completion(parsed=self._query, usage=self._usage)


def _case(sl_context: str = "Metrics:\n  revenue (simple)") -> MetricCase:
    return MetricCase(
        id="c",
        input="Total revenue?",
        gold=MetricQuery(metrics=["revenue"]),
        platform=PlatformRef(name="local", kind="duckdb"),
        target_dir="target",
        sl_context=sl_context,
    )


class TestMetricLayerSolver:
    def test_happy_path(self) -> None:
        query = MetricQuery(metrics=["revenue"], group_by=["metric_time__month"])
        out = MetricLayerSolver(model=StubLlm(query)).solve(_case())
        assert out.error is None
        assert out.query == query
        assert out.metadata["model"] == "StubLlm"

    def test_prompt_includes_context_and_question(self) -> None:
        stub = StubLlm(MetricQuery(metrics=["revenue"]))
        MetricLayerSolver(model=stub).solve(_case(sl_context="SL-CTX-MARKER"))
        assert "SL-CTX-MARKER" in stub.prompts[-1]
        assert "Total revenue?" in stub.prompts[-1]

    def test_malformed_output_is_invalid_structured_output(self) -> None:
        err = LlmError(kind="malformed_output", message="model returned malformed structured output")
        out = MetricLayerSolver(model=StubLlm(err)).solve(_case())
        assert out.query is None
        assert out.error is not None
        assert out.error.kind == "invalid_structured_output"

    @pytest.mark.parametrize(
        "kind",
        ["timeout", "rate_limit", "auth", "context_window_exceeded", "bad_request", "api_connection", "api_error"],
    )
    def test_provider_error_maps_one_to_one(self, kind: ProviderErrorKind) -> None:
        out = MetricLayerSolver(model=StubLlm(LlmError(kind=kind, message="boom", provider="openai"))).solve(_case())
        assert out.query is None
        assert out.error is not None
        assert out.error.kind == kind
        assert out.error.provider == "openai"

    def test_telemetry_passthrough(self) -> None:
        usage = Usage(prompt_tokens=11, completion_tokens=7, cost_usd=0.0003, latency_seconds=1.5)
        out = MetricLayerSolver(model=_FixedLlm(MetricQuery(metrics=["revenue"]), usage)).solve(_case())
        assert out.prompt_tokens == 11
        assert out.completion_tokens == 7
        assert out.cost_usd == 0.0003
        assert out.latency_seconds == 1.5

    def test_custom_template_rendered(self) -> None:
        stub = StubLlm(MetricQuery(metrics=["revenue"]))
        MetricLayerSolver(model=stub, prompt_template="LAYER={semantic_layer} Q={input}").solve(_case(sl_context="L"))
        assert stub.prompts[-1] == "LAYER=L Q=Total revenue?"

    def test_model_string_is_recorded(self) -> None:
        # The model-string path builds the litellm backend; no request is made at construction.
        assert MetricLayerSolver("openai/gpt-4o-mini")._model == "openai/gpt-4o-mini"

    def test_satisfies_solver_protocol(self) -> None:
        assert isinstance(MetricLayerSolver(model=StubLlm(MetricQuery(metrics=["revenue"]))), MetricSolver)


class TestMetricSolverOutput:
    def test_rejects_neither_query_nor_error(self) -> None:
        with pytest.raises(ValidationError):
            MetricSolverOutput()

    def test_rejects_both_query_and_error(self) -> None:
        with pytest.raises(ValidationError):
            MetricSolverOutput(query=MetricQuery(metrics=["revenue"]), error=SolverError(kind="auth", message="bad"))
