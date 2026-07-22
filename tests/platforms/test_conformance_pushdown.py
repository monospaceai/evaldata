"""Conformance battery for SQL pushdown: identical `passed`/`count` across every registered
platform adapter.

Each check builds its model from engine-portable inline SELECTs, runs the real
`ExpectationSuiteScorer` over a real `QueryRunner`, and asserts the same outcome on every
engine — proving the checks use engine-native semantics, not a Python re-implementation.
"""

import pytest

from evaldata.platforms.base import PlatformAdapter
from evaldata.scorers import ExpectationSuiteScorer, QueryRunner, ScoreContext
from evaldata.types import (
    Column,
    EvalCase,
    ExecutionSuccess,
    Expectation,
    ExpectationOutcome,
    ExpectationSuite,
    NotNullExpectation,
    PlatformKind,
    RowCountExpectation,
    SolverSuccess,
    Sql,
    SqlType,
    UniqueExpectation,
)

from .conftest import conform_name, engine_params, platform_ref, render_model


@pytest.fixture(params=engine_params())
def engine(request: pytest.FixtureRequest) -> tuple[PlatformAdapter, PlatformKind]:
    """An (adapter, dialect) pair, parametrised across every registered platform adapter."""
    return request.param()


def _outcome(
    engine: tuple[PlatformAdapter, PlatformKind],
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
    model_sql = Sql(render_model(model, dialect))
    case = EvalCase(
        id="c",
        input="q",
        expected=ExpectationSuite(expectations=[expectation]),
        platform=platform_ref(dialect),
    )
    queries = QueryRunner(adapter, model_sql, dialect, None)
    schema = [Column(name=name, type=SqlType.parse("INTEGER", dialect)) for name in (columns or [])] or None
    result = ExecutionSuccess(rows=[], schema=schema, latency_seconds=0.0)
    context = ScoreContext(queries=queries)
    score = ExpectationSuiteScorer().score(case, SolverSuccess(output=model_sql), result, context=context)
    assert len(score.outcomes) == 1
    return score.outcomes[0]


_THREE_ROWS = "SELECT 1 AS id UNION ALL SELECT 2 AS id UNION ALL SELECT 3 AS id"
_DISTINCT = "SELECT 1 AS id UNION ALL SELECT 2 AS id"
_WITH_DUP = "SELECT 1 AS id UNION ALL SELECT 1 AS id"
_WITH_NULLS = "SELECT CAST(NULL AS INTEGER) AS id UNION ALL SELECT CAST(NULL AS INTEGER) AS id UNION ALL SELECT 1 AS id"
_TWO_NULLS = "SELECT 'a' AS email UNION ALL SELECT NULL AS email UNION ALL SELECT NULL AS email"
_NO_NULLS = "SELECT 'a' AS email UNION ALL SELECT 'b' AS email"
_QUOTED_DUP = 'SELECT 1 AS "order" UNION ALL SELECT 1 AS "order"'


def test_row_count_pass(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    outcome = _outcome(engine, _THREE_ROWS, RowCountExpectation(exact=3))
    assert outcome.passed is True


def test_row_count_fail(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    outcome = _outcome(engine, _THREE_ROWS, RowCountExpectation(exact=5))
    assert outcome.passed is False
    assert outcome.actual == "3"


def test_not_null_pass(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    email = conform_name("email", dialect)
    outcome = _outcome(engine, _NO_NULLS, NotNullExpectation(column=email), columns=[email])
    assert outcome.passed is True
    assert outcome.count == 0


def test_not_null_fail_counts_and_samples(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    email = conform_name("email", dialect)
    outcome = _outcome(engine, _TWO_NULLS, NotNullExpectation(column=email), columns=[email])
    assert outcome.passed is False
    assert outcome.count == 2
    assert outcome.sample_rows == [{email: None}, {email: None}]


def test_unique_pass(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    _, dialect = engine
    id_ = conform_name("id", dialect)
    outcome = _outcome(engine, _DISTINCT, UniqueExpectation(column=id_), columns=[id_])
    assert outcome.passed is True
    assert outcome.count == 0


def test_unique_fail_counts_and_samples(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # `unique_sample`'s count column is a hardcoded, unquoted `n` alias, so it folds too.
    _, dialect = engine
    id_ = conform_name("id", dialect)
    n = conform_name("n", dialect)
    outcome = _outcome(engine, _WITH_DUP, UniqueExpectation(column=id_), columns=[id_])
    assert outcome.passed is False
    assert outcome.count == 1
    assert outcome.sample_rows == [{id_: 1, n: 2}]


def test_unique_excludes_nulls(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # Two NULLs plus one distinct non-NULL: NULLs are excluded, so unique passes.
    _, dialect = engine
    id_ = conform_name("id", dialect)
    outcome = _outcome(engine, _WITH_NULLS, UniqueExpectation(column=id_), columns=[id_])
    assert outcome.passed is True
    assert outcome.count == 0


def test_unique_quoted_identifier_column(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # A column named `order` (reserved) with a duplicate fails — proving sqlglot quoting.
    # The model quotes "order" explicitly, so no dialect folds it; stays literal.
    outcome = _outcome(engine, _QUOTED_DUP, UniqueExpectation(column="order"), columns=["order"])
    assert outcome.passed is False
    assert outcome.count == 1


def test_derived_query_error_fails_outcome(engine: tuple[PlatformAdapter, PlatformKind]) -> None:
    # A model referencing a missing table makes the derived not_null query error; the
    # outcome fails (errors-as-values) rather than raising.
    _, dialect = engine
    email = conform_name("email", dialect)
    outcome = _outcome(
        engine, "SELECT email FROM does_not_exist_xyz", NotNullExpectation(column=email), columns=[email]
    )
    assert outcome.passed is False
    assert outcome.detail is not None
