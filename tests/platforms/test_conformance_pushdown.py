"""Conformance battery for SQL pushdown: identical `passed`/`count` on DuckDB and Postgres.

Each check builds its model from engine-portable inline SELECTs, runs the real
`ExpectationSuiteScorer` over a real `QueryRunner`, and asserts the same outcome on both
engines — proving the checks use engine-native semantics, not a Python re-implementation.
"""

import pytest

from dataeval.platforms.base import PlatformAdapter
from dataeval.platforms.duckdb import DuckDBAdapter
from dataeval.scorers import ExpectationSuiteScorer, QueryRunner, ScoreContext
from dataeval.scorers.sql import Dialect
from dataeval.types import (
    Column,
    EvalCase,
    ExecutionResult,
    Expectation,
    ExpectationOutcome,
    ExpectationSuite,
    NotNullExpectation,
    PlatformRef,
    RowCountExpectation,
    SolverOutput,
    Sql,
    SqlType,
    UniqueExpectation,
)

from .conftest import connect_postgres_or_skip


def _duckdb_engine() -> tuple[PlatformAdapter, Dialect]:
    return DuckDBAdapter(), "duckdb"


def _postgres_engine() -> tuple[PlatformAdapter, Dialect]:
    return connect_postgres_or_skip(), "postgres"


@pytest.fixture(
    params=[
        pytest.param(_duckdb_engine, id="duckdb", marks=pytest.mark.unit),
        pytest.param(_postgres_engine, id="postgres", marks=pytest.mark.e2e),
    ],
)
def engine(request: pytest.FixtureRequest) -> tuple[PlatformAdapter, Dialect]:
    """An (adapter, dialect) pair, parametrised across DuckDB (unit) and Postgres (e2e)."""
    return request.param()


def _outcome(
    engine: tuple[PlatformAdapter, Dialect],
    model: str,
    expectation: Expectation,
    *,
    columns: list[str] | None = None,
) -> ExpectationOutcome:
    """Score a single-expectation suite over `model` and return the sole outcome.

    `columns` names the model result's columns so the not_null/unique absent-column guards
    (which read the model result's schema, not the warehouse) see the checked column.
    """
    adapter, dialect = engine
    model_sql = Sql(model)
    case = EvalCase(
        id="c",
        input="q",
        expected=ExpectationSuite(expectations=[expectation]),
        platform=PlatformRef(name="x", kind="postgres" if dialect == "postgres" else "duckdb"),
    )
    queries = QueryRunner(adapter, model_sql, dialect, None)
    schema = [Column(name=name, type=SqlType.parse("INTEGER", dialect)) for name in (columns or [])] or None
    result = ExecutionResult(rows=[], schema=schema, latency_seconds=0.0)
    context = ScoreContext(queries=queries)
    score = ExpectationSuiteScorer().score(case, SolverOutput(output=model_sql), result, context=context)
    assert len(score.outcomes) == 1
    return score.outcomes[0]


_THREE_ROWS = "SELECT 1 AS id UNION ALL SELECT 2 AS id UNION ALL SELECT 3 AS id"
_DISTINCT = "SELECT 1 AS id UNION ALL SELECT 2 AS id"
_WITH_DUP = "SELECT 1 AS id UNION ALL SELECT 1 AS id"
_WITH_NULLS = "SELECT CAST(NULL AS INTEGER) AS id UNION ALL SELECT CAST(NULL AS INTEGER) AS id UNION ALL SELECT 1 AS id"
_TWO_NULLS = "SELECT 'a' AS email UNION ALL SELECT NULL AS email UNION ALL SELECT NULL AS email"
_NO_NULLS = "SELECT 'a' AS email UNION ALL SELECT 'b' AS email"
_QUOTED_DUP = 'SELECT 1 AS "order" UNION ALL SELECT 1 AS "order"'


def test_row_count_pass(engine: tuple[PlatformAdapter, Dialect]) -> None:
    outcome = _outcome(engine, _THREE_ROWS, RowCountExpectation(exact=3))
    assert outcome.passed is True


def test_row_count_fail(engine: tuple[PlatformAdapter, Dialect]) -> None:
    outcome = _outcome(engine, _THREE_ROWS, RowCountExpectation(exact=5))
    assert outcome.passed is False
    assert outcome.actual == "3"


def test_not_null_pass(engine: tuple[PlatformAdapter, Dialect]) -> None:
    outcome = _outcome(engine, _NO_NULLS, NotNullExpectation(column="email"), columns=["email"])
    assert outcome.passed is True
    assert outcome.count == 0


def test_not_null_fail_counts_and_samples(engine: tuple[PlatformAdapter, Dialect]) -> None:
    outcome = _outcome(engine, _TWO_NULLS, NotNullExpectation(column="email"), columns=["email"])
    assert outcome.passed is False
    assert outcome.count == 2
    assert outcome.sample_rows == [{"email": None}, {"email": None}]


def test_unique_pass(engine: tuple[PlatformAdapter, Dialect]) -> None:
    outcome = _outcome(engine, _DISTINCT, UniqueExpectation(column="id"), columns=["id"])
    assert outcome.passed is True
    assert outcome.count == 0


def test_unique_fail_counts_and_samples(engine: tuple[PlatformAdapter, Dialect]) -> None:
    outcome = _outcome(engine, _WITH_DUP, UniqueExpectation(column="id"), columns=["id"])
    assert outcome.passed is False
    assert outcome.count == 1
    assert outcome.sample_rows == [{"id": 1, "n": 2}]


def test_unique_excludes_nulls(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # Two NULLs plus one distinct non-NULL: NULLs are excluded, so unique passes.
    outcome = _outcome(engine, _WITH_NULLS, UniqueExpectation(column="id"), columns=["id"])
    assert outcome.passed is True
    assert outcome.count == 0


def test_unique_quoted_identifier_column(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # A column named `order` (reserved) with a duplicate fails — proving sqlglot quoting.
    outcome = _outcome(engine, _QUOTED_DUP, UniqueExpectation(column="order"), columns=["order"])
    assert outcome.passed is False
    assert outcome.count == 1


def test_derived_query_error_fails_outcome(engine: tuple[PlatformAdapter, Dialect]) -> None:
    # A model referencing a missing table makes the derived not_null query error; the
    # outcome fails (errors-as-values) rather than raising.
    outcome = _outcome(
        engine, "SELECT email FROM does_not_exist_xyz", NotNullExpectation(column="email"), columns=["email"]
    )
    assert outcome.passed is False
    assert outcome.detail is not None
