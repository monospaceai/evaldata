"""Tests for `CallableSolver` — the hermetic no-LLM solver."""

import pytest

from evaldata.solvers import CallableSolver, Solver
from evaldata.types import DuckDBPlatformRef, EvalCase, SolverSuccess, UntypedResultSet


def _case() -> EvalCase:
    return EvalCase(
        id="c",
        input="q",
        expected=UntypedResultSet(rows=[]),
        platform=DuckDBPlatformRef(name="x"),
    )


@pytest.mark.unit
class TestCallableSolver:
    def test_wraps_sql_string_as_output(self) -> None:
        out = CallableSolver(lambda case: "SELECT 1").solve(_case())
        assert isinstance(out, SolverSuccess)
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
