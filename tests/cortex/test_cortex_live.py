from decimal import Decimal
from typing import Any

import pytest

from evaldata.cortex.client import CortexAnalystClient
from evaldata.cortex.solver import CortexAnalystSolver
from evaldata.platforms.base import PlatformAdapter
from evaldata.types import EvalCase, ExecutionSuccess, SnowflakeConfig, SnowflakePlatformRef, SolverSuccess

pytestmark = [pytest.mark.cortex, pytest.mark.e2e, pytest.mark.manual]


def _cell(row: dict[str, Any], needle: str) -> Any:
    for key, value in row.items():
        if needle in key.upper():
            return value
    msg = f"no column matching {needle!r} in {list(row)}"
    raise KeyError(msg)


def test_cortex_answers_region_totals(live_adapter: PlatformAdapter, jaffle_view: str) -> None:
    solver = CortexAnalystSolver(
        CortexAnalystClient.from_connection(live_adapter.connection), semantic_view=jaffle_view
    )
    case = EvalCase(
        id="region-totals",
        input="What is the total order amount for each customer region?",
        expected={"rows": []},
        platform=SnowflakePlatformRef(name="sf", config=SnowflakeConfig(account="test")),
    )

    output = solver.solve(case)
    assert isinstance(output, SolverSuccess)

    result = live_adapter.execute(output.output)
    assert isinstance(result, ExecutionSuccess)
    totals = {_cell(row, "REGION"): _cell(row, "AMOUNT") for row in result.rows}
    assert totals == {"East": Decimal("185.50"), "West": Decimal("380.00")}
