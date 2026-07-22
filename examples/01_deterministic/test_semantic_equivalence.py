"""Semantic-equivalence example evals: AI SQL that differs syntactically but is semantically equivalent.

`SemanticEquivalence` compares the queries and confirms a match without running them;
`observed_equivalence()` adds a fallback that runs both queries and compares their results.
"""

import tempfile
from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from evaldata import CallableSolver, EvalCase, SemanticEquivalence, eval_case, observed_equivalence
from evaldata.platforms import duckdb_platform, resolve
from evaldata.scorers import AstEquivalence, QueryRunner, ScoreContext, Scorer
from evaldata.types import ScoreResult, SolverSuccess

_DB_PATH = Path(tempfile.mkdtemp(prefix="evaldata_ex01_sem_")) / "shop.duckdb"
_PLATFORM = duckdb_platform(name="examples-semantic-equivalence", path=str(_DB_PATH))


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
        scorer: The scorer to run: `SemanticEquivalence` (compares the queries) or composite `observed_equivalence`.

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
    """The `(scorer, passed)` of each member that ran under `observed_equivalence`, in order.

    Args:
        result: An `observed_equivalence` score result.

    Returns:
        One `(scorer, passed)` pair per member that ran.
    """
    return [(entry["scorer"], entry["passed"]) for entry in result.metadata["first_decisive"]]


def _verdicts(result: ScoreResult) -> list[tuple[str, str]]:
    """The `(method, equivalence)` of each verdict a `SemanticEquivalence` result recorded.

    Args:
        result: A `SemanticEquivalence` score result.

    Returns:
        One `(method, equivalence)` pair per check that ran.
    """
    return [(v["method"], v["equivalence"]) for v in result.metadata["verdicts"]]


@eval_case(
    input="Which US customers have an id above 1?",
    expected={"kind": "gold_query", "sql": "SELECT name FROM customers WHERE country = 'US' AND id > 1"},
    platform=_PLATFORM,
)
def test_ast_confirms_without_executing(case: EvalCase) -> None:
    """Reordered predicates and casing match after normalization; confirmed without running either query."""
    result = _score(case, "select NAME from customers where id > 1 and country = 'US'", observed_equivalence())
    assert result.passed
    # A structural confirmation skips execution, so only the first member ran.
    assert _trail(result) == [("semantic_equivalence", True)]


@eval_case(
    input="Name the US customers.",
    expected={"kind": "gold_query", "sql": "SELECT name FROM customers WHERE country = 'US'"},
    platform=_PLATFORM,
)
def test_execution_confirms_when_ast_is_inconclusive(case: EvalCase) -> None:
    """A CTE the syntax check can't match; the execution fallback runs both queries and confirms."""
    result = _score(
        case, "WITH us AS (SELECT * FROM customers WHERE country = 'US') SELECT name FROM us", observed_equivalence()
    )
    assert result.passed
    assert _trail(result) == [("semantic_equivalence", False), ("result_set_equivalence", True)]


@eval_case(
    input="Name the US customers.",
    expected={"kind": "gold_query", "sql": "SELECT name FROM customers WHERE country = 'US'"},
    platform=_PLATFORM,
)
def test_execution_refutes_wrong_query(case: EvalCase) -> None:
    """A wrong filter; the syntax check is inconclusive and execution refutes with a diff."""
    result = _score(case, "SELECT name FROM customers WHERE country = 'GB'", observed_equivalence())
    assert not result.passed
    assert _trail(result) == [("semantic_equivalence", False), ("result_set_equivalence", False)]
    assert result.diff is not None
    assert result.diff.missing_row_count == 2  # 'Bo', 'Cy' expected but absent
    assert result.diff.extra_row_count == 1  # 'Ada' returned but not expected


@eval_case(
    input="What time is it now?",
    expected={"kind": "gold_query", "sql": "SELECT current_timestamp AS t"},
    platform=_PLATFORM,
)
def test_ast_inconclusive_on_nondeterminism(case: EvalCase) -> None:
    """`current_timestamp` can't be compared on syntax; `SemanticEquivalence` is inconclusive."""
    result = _score(case, "SELECT current_timestamp AS t", SemanticEquivalence([AstEquivalence()]))
    assert not result.passed
    assert result.verdict == "inconclusive"
    assert result.explanation == "no semantic check could confirm equivalence"
    assert _verdicts(result) == [("ast", "unknown")]
