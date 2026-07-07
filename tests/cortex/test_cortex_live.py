"""Live end-to-end test against the real Cortex Analyst endpoint (costs Snowflake credits).

Marked `cortex`: excluded from the PR gate and the default local check, run manually or by a
dedicated off-by-default job. Builds the jaffle-shop fixture, asks Cortex Analyst a question,
executes the SQL it returns, and asserts the answer matches the known totals.
"""

from decimal import Decimal
from typing import Any

import pytest

from evaldata.cortex.client import CortexAnalystClient
from evaldata.cortex.solver import CortexAnalystSolver
from evaldata.platforms.base import PlatformAdapter
from evaldata.types import EvalCase, PlatformRef

pytestmark = [pytest.mark.cortex, pytest.mark.e2e]


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
        platform=PlatformRef(name="sf", kind="snowflake"),
    )

    output = solver.solve(case)
    assert output.error is None, output.error
    assert output.output is not None

    result = live_adapter.execute(output.output)
    assert result.error is None, result.error
    totals = {_cell(row, "REGION"): _cell(row, "AMOUNT") for row in result.rows}
    assert totals == {"East": Decimal("185.50"), "West": Decimal("380.00")}
