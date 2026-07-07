"""Unit tests for `CortexAnalystSolver` — parsing, error mapping, and reference validation."""

from typing import Any

import pytest

from evaldata.cortex.solver import CortexAnalystSolver
from evaldata.types import EvalCase, PlatformRef, SolverError


class _FakeTransport:
    """A `CortexTransport` returning a fixed reply and recording each call."""

    def __init__(self, reply: "dict[str, Any] | SolverError") -> None:
        self._reply = reply
        self.calls: list[tuple[str, dict[str, str]]] = []

    def send(self, question: str, semantic_ref: dict[str, str]) -> "dict[str, Any] | SolverError":
        self.calls.append((question, semantic_ref))
        return self._reply


def _case(question: str = "What is the total order amount by region?") -> EvalCase:
    return EvalCase(
        id="q1",
        input=question,
        expected={"rows": []},
        platform=PlatformRef(name="sf", kind="snowflake"),
    )


def _sql_reply(statement: str) -> dict[str, Any]:
    return {
        "request_id": "req-123",
        "message": {
            "role": "analyst",
            "content": [{"type": "text", "text": "interpretation"}, {"type": "sql", "statement": statement}],
        },
        "response_metadata": {"model_names": ["claude-sonnet-4-5"]},
    }


@pytest.mark.unit
class TestReferenceValidation:
    def test_accepts_a_semantic_view(self) -> None:
        solver = CortexAnalystSolver(_FakeTransport(_sql_reply("SELECT 1")), semantic_view="DB.S.V")
        assert solver._semantic_ref == {"semantic_view": "DB.S.V"}  # noqa: SLF001

    def test_accepts_a_semantic_model_file(self) -> None:
        solver = CortexAnalystSolver(_FakeTransport(_sql_reply("SELECT 1")), semantic_model_file="@db.s.stage/m.yaml")
        assert solver._semantic_ref == {"semantic_model_file": "@db.s.stage/m.yaml"}  # noqa: SLF001

    def test_rejects_neither(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            CortexAnalystSolver(_FakeTransport(_sql_reply("SELECT 1")))

    def test_rejects_both(self) -> None:
        with pytest.raises(ValueError, match="exactly one of"):
            CortexAnalystSolver(
                _FakeTransport(_sql_reply("SELECT 1")), semantic_view="V", semantic_model_file="@s/m.yaml"
            )


@pytest.mark.unit
class TestSolve:
    def _solver(self, reply: "dict[str, Any] | SolverError") -> CortexAnalystSolver:
        return CortexAnalystSolver(_FakeTransport(reply), semantic_view="DB.S.V")

    def test_returns_generated_sql_with_telemetry(self) -> None:
        transport = _FakeTransport(_sql_reply("SELECT region FROM t"))
        solver = CortexAnalystSolver(transport, semantic_view="DB.S.V")
        output = solver.solve(_case("q?"))

        assert output.output == "SELECT region FROM t"
        assert output.error is None
        assert output.metadata == {"request_id": "req-123", "model_names": ["claude-sonnet-4-5"]}
        assert transport.calls == [("q?", {"semantic_view": "DB.S.V"})]

    def test_suggestions_become_an_empty_response_error(self) -> None:
        reply = {
            "request_id": "req-9",
            "message": {"content": [{"type": "suggestions", "suggestions": ["Do you mean revenue?", "By month?"]}]},
        }
        output = self._solver(reply).solve(_case())
        assert output.output is None
        assert isinstance(output.error, SolverError)
        assert output.error.kind == "empty_response"
        assert "Do you mean revenue?; By month?" in output.error.message
        assert output.metadata == {"request_id": "req-9"}

    def test_no_content_becomes_an_empty_response_error(self) -> None:
        output = self._solver({"message": {"content": []}}).solve(_case())
        assert isinstance(output.error, SolverError)
        assert output.error.kind == "empty_response"
        assert output.error.message == "Cortex Analyst returned no SQL"

    def test_blank_sql_statement_is_treated_as_no_sql(self) -> None:
        output = self._solver(_sql_reply("   ")).solve(_case())
        assert output.output is None
        assert isinstance(output.error, SolverError)
        assert output.error.kind == "empty_response"

    def test_transport_error_passes_through(self) -> None:
        error = SolverError(kind="auth", message="401", provider="cortex_analyst")
        output = self._solver(error).solve(_case())
        assert output.output is None
        assert output.error is error
