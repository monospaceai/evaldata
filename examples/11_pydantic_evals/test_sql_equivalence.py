"""Pydantic Evals integration with execution-based SQL scoring on DuckDB."""

from collections.abc import Iterator

import pytest

from evaldata.platforms import duckdb_platform, resolve
from evaldata.pydantic_evals import SqlEquivalence, close_all
from evaldata.types import ExecutionSuccess
from pydantic_evals import Case, Dataset

_PLATFORM = duckdb_platform(name="examples-pydantic-evals")
_GENERATED_SQL = {
    "What is the total order amount by region?": (
        "WITH totals AS ("
        "SELECT region, SUM(amount) AS total FROM orders GROUP BY region"
        ") SELECT region, total FROM totals"
    ),
    "What is the average order amount by region?": (
        "SELECT region, COUNT(amount) AS average_amount FROM orders GROUP BY region"
    ),
}


@pytest.fixture(scope="module", autouse=True)
def _database() -> Iterator[None]:
    result = resolve(_PLATFORM).execute(
        "CREATE TABLE orders (id INTEGER, region VARCHAR, amount INTEGER); "
        "INSERT INTO orders VALUES (1, 'east', 50), (2, 'west', 120), (3, 'east', 20)"
    )
    assert isinstance(result, ExecutionSuccess)
    try:
        yield
    finally:
        close_all()


def generate_sql(question: str) -> str:
    """Return the fixed generated SQL for a question."""
    return _GENERATED_SQL[question]


def test_sql_equivalence() -> None:
    """Score correct and incorrect generated SQL through Pydantic Evals."""
    dataset = Dataset(
        name="text-to-sql",
        cases=[
            Case(
                name="totals-by-region",
                inputs="What is the total order amount by region?",
                expected_output="SELECT region, SUM(amount) AS total FROM orders GROUP BY region",
            ),
            Case(
                name="average-by-region",
                inputs="What is the average order amount by region?",
                expected_output=("SELECT region, AVG(amount) AS average_amount FROM orders GROUP BY region"),
            ),
        ],
        evaluators=[SqlEquivalence(platform=_PLATFORM)],
    )

    report = dataset.evaluate_sync(generate_sql)
    verdicts = {case.name: case.assertions["SqlEquivalence"].value for case in report.cases}

    assert verdicts == {"totals-by-region": True, "average-by-region": False}
