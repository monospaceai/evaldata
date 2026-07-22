"""LLM-judge example evals: `judged_equivalence` has an LLM grade the SQL the syntax check can't confirm."""

import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from evaldata import CallableSolver, EvalCase, eval_case, judged_equivalence
from evaldata.platforms import duckdb_platform, resolve
from evaldata.scorers import QueryRunner, ScoreContext, Scorer
from evaldata.types import ScoreResult, SolverSuccess

_DB_PATH = Path(tempfile.mkdtemp(prefix="evaldata_ex05_")) / "shop.duckdb"
_PLATFORM = duckdb_platform(name="examples-llm-judge", path=str(_DB_PATH))
_MODEL = os.getenv("EVALDATA_HOSTED_MODEL", "openai/gpt-4o-mini")


@pytest.fixture(scope="module", autouse=True)
def _seed_db() -> Iterator[None]:
    con = duckdb.connect(str(_DB_PATH))
    con.execute("CREATE TABLE customers (id INTEGER, name VARCHAR, country VARCHAR)")
    con.execute("INSERT INTO customers VALUES (1, 'Ada', 'GB'), (2, 'Bo', 'US'), (3, 'Cy', 'US')")
    con.close()
    yield


def _score(case: EvalCase, model_sql: str, scorer: Scorer) -> ScoreResult:
    """Score `model_sql` against the case's gold query with `scorer`.

    Args:
        case: The eval case, carrying the gold query and platform.
        model_sql: The model's SQL to judge against the gold query.
        scorer: The scorer to run.

    Returns:
        The `ScoreResult`.
    """
    solver = CallableSolver(lambda _case: model_sql)
    output = solver.solve(case)
    assert isinstance(output, SolverSuccess)
    sql = output.output
    dialect = case.platform.dialect or case.platform.kind
    runner = QueryRunner(resolve(case.platform), sql, dialect, None)
    context = ScoreContext(queries=runner)
    return scorer.score(case, output, runner.run(sql), context=context)


def _trail(result: ScoreResult) -> list[tuple[str, bool]]:
    """The `(scorer, passed)` of each member that ran, in order.

    Args:
        result: A `judged_equivalence` score result.

    Returns:
        One `(scorer, passed)` pair per member that ran.
    """
    return [(entry["scorer"], entry["passed"]) for entry in result.metadata["first_decisive"]]


@eval_case(
    input="Name the US customers.",
    expected={"kind": "gold_query", "sql": "SELECT name FROM customers WHERE country = 'US'"},
    platform=_PLATFORM,
)
def test_judge_confirms_when_ast_is_inconclusive(case: EvalCase) -> None:
    """A CTE the syntax check can't match; the judge confirms it without running either query."""
    result = _score(
        case,
        "WITH us AS (SELECT * FROM customers WHERE country = 'US') SELECT name FROM us",
        judged_equivalence(_MODEL),
    )
    assert result.passed
    assert _trail(result) == [("semantic_equivalence", False), ("llm_judge", True)]


@eval_case(
    input="Name the US customers.",
    expected={"kind": "gold_query", "sql": "SELECT name FROM customers WHERE country = 'US'"},
    platform=_PLATFORM,
)
def test_judge_refutes_wrong_filter(case: EvalCase) -> None:
    """A wrong filter the syntax check can't match; the judge refutes it without running either query."""
    result = _score(
        case,
        "WITH gb AS (SELECT * FROM customers WHERE country = 'GB') SELECT name FROM gb",
        judged_equivalence(_MODEL),
    )
    assert not result.passed
    assert _trail(result) == [("semantic_equivalence", False), ("llm_judge", False)]
