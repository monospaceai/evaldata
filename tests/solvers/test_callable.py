"""Tests for `CallableSolver` — the hermetic no-LLM solver."""

import pytest

from dataeval.solvers import CallableSolver, Solver
from dataeval.types import EvalCase, PlatformRef, SolverOutput, UntypedResultSet


def _case() -> EvalCase:
    return EvalCase(
        id="c",
        input="q",
        expected=UntypedResultSet(rows=[]),
        platform=PlatformRef(name="x", kind="duckdb"),
    )


@pytest.mark.unit
class TestCallableSolver:
    def test_wraps_sql_string_as_output(self) -> None:
        out = CallableSolver(lambda case: "SELECT 1").solve(_case())
        assert isinstance(out, SolverOutput)
        assert out.output == "SELECT 1"

    def test_function_receives_the_case(self) -> None:
        seen: dict[str, str] = {}

        def fn(case: EvalCase) -> str:
            seen["id"] = case.id
            return "SELECT 1"

        CallableSolver(fn).solve(_case())
        assert seen["id"] == "c"

    def test_satisfies_solver_protocol(self) -> None:
        assert isinstance(CallableSolver(lambda case: "SELECT 1"), Solver)
