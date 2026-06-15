"""Conformance plumbing: `ConformanceFixtures` Protocol, per-adapter fixture types, and the parametrised `under_test` fixture."""

import os
from dataclasses import dataclass
from typing import Protocol

import pytest

from dataeval.platforms.base import PlatformAdapter
from dataeval.platforms.duckdb import DuckDBAdapter


class ConformanceFixtures(Protocol):
    """The shared vocabulary of SQL fragments every adapter must supply.

    Each attribute names a *behaviour*; its value is the adapter's dialect-specific
    SQL that exercises it. Adapters implement this Protocol structurally via their
    own concrete `@dataclass(frozen=True)` types.
    """

    one_row_one_column: str  # returns one row with one column named "n"
    empty_result: str  # returns zero rows with a known schema (column "n")
    three_rows: str  # returns three rows in column "n" with values 1, 2, 3
    null_value: str  # returns one row with NULL in column "x"
    duplicate_column_names: str  # returns two columns both named "x"
    references_missing_table: str  # references a non-existent table
    parse_error: str  # is syntactically invalid
    slow_query: str  # runs long enough to overrun a sub-second budget, and is interruptible


@dataclass(frozen=True)
class DuckDBFixtures:
    """DuckDB's concrete `ConformanceFixtures` — structurally satisfies the Protocol."""

    one_row_one_column: str = "SELECT 1 AS n"
    empty_result: str = "SELECT 1 AS n WHERE 1=0"
    three_rows: str = "SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3"
    null_value: str = "SELECT NULL AS x"
    duplicate_column_names: str = "SELECT 1 AS x, 2 AS x"
    references_missing_table: str = "SELECT * FROM does_not_exist_xyz"
    parse_error: str = "SELECT FROM nope"
    # No pg_sleep equivalent: a recursive CTE counting to 100M spins long enough to overrun
    # a sub-second budget, and DuckDB checks for interrupts between iterations.
    slow_query: str = (
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 100000000) "
        "SELECT count(*) AS n FROM t"
    )


@dataclass(frozen=True)
class PostgresFixtures:
    """PostgreSQL's concrete `ConformanceFixtures` — idiomatic Postgres SQL.

    Diverges from DuckDB where the idiom differs: `VALUES` row lists, real
    boolean literals (`WHERE false`), and a genuine syntax error for
    `parse_error` (`SELECT FROM nope` is only a *missing-relation* error in
    Postgres, not a parse error — it would duplicate `references_missing_table`).
    """

    one_row_one_column: str = "SELECT 1 AS n"
    empty_result: str = "SELECT 1 AS n WHERE false"
    three_rows: str = "SELECT n FROM (VALUES (1), (2), (3)) AS t(n)"
    null_value: str = "SELECT NULL AS x"
    duplicate_column_names: str = "SELECT 1 AS x, 2 AS x"
    references_missing_table: str = "SELECT * FROM does_not_exist_xyz"
    parse_error: str = "SLECT 1"
    slow_query: str = "SELECT pg_sleep(10)"


@dataclass(frozen=True)
class UnderTest:
    """One adapter-under-test: its live `PlatformAdapter` + the SQL it uses."""

    adapter: PlatformAdapter
    fixtures: ConformanceFixtures


def _duckdb_under_test() -> UnderTest:
    return UnderTest(adapter=DuckDBAdapter(), fixtures=DuckDBFixtures())


def _postgres_dsn() -> str:
    """Assemble a libpq connection string from dbt-style `POSTGRES_TEST_*` env vars.

    Defaults match the bundled `docker-compose.yml` so a bare `docker compose up`
    needs no further configuration; CI overrides them to point at a service container.
    """
    host = os.environ.get("POSTGRES_TEST_HOST", "localhost")
    port = os.environ.get("POSTGRES_TEST_PORT", "5432")
    user = os.environ.get("POSTGRES_TEST_USER", "postgres")
    password = os.environ.get("POSTGRES_TEST_PASS", "postgres")
    dbname = os.environ.get("POSTGRES_TEST_DBNAME", "postgres")
    return f"host={host} port={port} user={user} password={password} dbname={dbname}"


def connect_postgres_or_skip() -> PlatformAdapter:
    """Connect a `PostgresAdapter` to the configured test database, or skip the test.

    Skips (rather than fails) when the `postgres` extra is not installed or no
    Postgres is reachable — keeping `pytest` green for contributors without one,
    while CI runs these via `-m e2e` against a service container. Shared by the
    `postgres` conformance param and `test_postgres.py`'s native-type tests.
    """
    try:
        from dataeval.platforms.postgres import PostgresAdapter
    except ImportError:
        pytest.skip("psycopg not installed; install the 'postgres' extra (uv sync --extra postgres)")
    import psycopg

    try:
        return PostgresAdapter(_postgres_dsn())
    except psycopg.OperationalError as e:
        pytest.skip(f"Postgres not reachable ({_postgres_dsn()}): {e}".strip())


def _postgres_under_test() -> UnderTest:
    return UnderTest(adapter=connect_postgres_or_skip(), fixtures=PostgresFixtures())


# Function-scoped: each test gets a fresh adapter.
@pytest.fixture(
    params=[
        pytest.param(_duckdb_under_test, id="duckdb", marks=pytest.mark.unit),
        pytest.param(_postgres_under_test, id="postgres", marks=pytest.mark.e2e),
    ],
)
def under_test(request: pytest.FixtureRequest) -> UnderTest:
    """Return one (adapter, fixtures) pair; parametrised across all registered adapters."""
    factory = request.param
    return factory()
