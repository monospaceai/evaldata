"""Live Cortex Analyst accuracy benchmark over the jaffle-shop semantic view (costs credits);
excluded from the default run via the `cortex` marker.
"""

import pytest

from evaldata import ExecutionAccuracy
from evaldata.core.runner import evaluate_case
from evaldata.cortex.client import CortexAnalystClient
from evaldata.cortex.solver import CortexAnalystSolver
from evaldata.platforms.base import PlatformAdapter
from evaldata.types import EvalCase, GoldQuery, SnowflakeConfig, SnowflakePlatformRef

pytestmark = [pytest.mark.cortex, pytest.mark.e2e]

_PLATFORM = SnowflakePlatformRef(name="jaffle", config=SnowflakeConfig(account="test"))
_ORDERS = "JAFFLE_SHOP_DB.PUBLIC.ORDERS"
_CUSTOMERS = "JAFFLE_SHOP_DB.PUBLIC.CUSTOMERS"
_JOIN = f"{_ORDERS} o JOIN {_CUSTOMERS} c ON o.CUSTOMER_ID = c.ID"

_CASES = [
    (
        "total-amount-by-region",
        "What is the total order amount for each customer region?",
        f"SELECT c.REGION, SUM(o.AMOUNT) FROM {_JOIN} GROUP BY c.REGION",
    ),
    (
        "orders-by-region",
        "How many orders were placed in each customer region?",
        f"SELECT c.REGION, COUNT(*) FROM {_JOIN} GROUP BY c.REGION",
    ),
    ("total-orders", "How many orders are there in total?", f"SELECT COUNT(*) FROM {_ORDERS}"),
    ("total-amount", "What is the total order amount across all orders?", f"SELECT SUM(AMOUNT) FROM {_ORDERS}"),
    (
        "customers-per-region",
        "How many customers are in each region?",
        f"SELECT REGION, COUNT(*) FROM {_CUSTOMERS} GROUP BY REGION",
    ),
]


def _case(case_id: str, question: str, gold_sql: str) -> EvalCase:
    return EvalCase(id=case_id, input=question, expected=GoldQuery(sql=gold_sql), platform=_PLATFORM)


def test_cortex_jaffle_benchmark(live_adapter: PlatformAdapter, jaffle_view: str) -> None:
    solver = CortexAnalystSolver(
        CortexAnalystClient.from_connection(live_adapter.connection), semantic_view=jaffle_view
    )
    scorer = ExecutionAccuracy(row_order="ignore", column_alignment="by_value")

    reports = [evaluate_case(_case(*row), solver, scorers=[scorer], adapter=live_adapter).report for row in _CASES]
    passed = sum(1 for report in reports if report.passed)
    accuracy = passed / len(reports)

    print(f"\nCortex Analyst jaffle benchmark: {accuracy:.0%} ({passed}/{len(reports)})")
    for report in reports:
        print(f"  {'PASS' if report.passed else 'FAIL'}  {report.id}")
    assert accuracy == 1.0, [(report.id, report.passed) for report in reports]
