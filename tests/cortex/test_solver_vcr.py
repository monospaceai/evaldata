"""Integration test for `CortexAnalystSolver` replayed from a recorded cassette."""

import pytest

from evaldata.cortex.solver import CortexAnalystSolver
from evaldata.types import EvalCase, SnowflakeConfig, SnowflakePlatformRef, SolverSuccess

_SEMANTIC_VIEW = "JAFFLE_SHOP_DB.PUBLIC.JAFFLE_SHOP_SV"


@pytest.mark.vcr
def test_solver_returns_sql_from_cortex(cortex_vcr_client: object) -> None:
    solver = CortexAnalystSolver(cortex_vcr_client, semantic_view=_SEMANTIC_VIEW)  # type: ignore[arg-type]
    case = EvalCase(
        id="region-totals",
        input="What is the total order amount for each customer region?",
        expected={"rows": []},
        platform=SnowflakePlatformRef(name="sf", config=SnowflakeConfig(account="test")),
    )

    output = solver.solve(case)

    assert isinstance(output, SolverSuccess)
    assert "SEMANTIC_VIEW" in output.output.upper()
    assert output.metadata.get("request_id")
