"""Conformance plumbing: `ConformanceFixtures` Protocol, per-adapter fixture types, and the parametrised `under_test` fixture."""

import os
from dataclasses import dataclass
from typing import Protocol

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.duckdb import DuckDBAdapter
from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.scorers.sql import Dialect


def conform_name(name: str, dialect: Dialect) -> str:
    """Return `name` as `dialect` folds an unquoted identifier (e.g. UPPERCASE on Snowflake)."""
    return normalize_identifiers(exp.to_identifier(name), dialect=dialect).name


def render_model(base: str, dialect: Dialect) -> str:
    """Render a conformance model query, authored in Postgres SQL, for `dialect`.

    Conformance models are executed raw, not through the scorer's renderer, so engine syntax
    divergences are transpiled per dialect from one authored base. Divergences transpilation
    can't infer (a bare `NUMERIC`'s scale) are authored per dialect at the call site instead.
    """
    return sqlglot.transpile(base, read="postgres", write=dialect)[0]


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
    reports_types: bool  # whether the driver reports result-column types (False for SQLite)


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
    reports_types: bool = True


@dataclass(frozen=True)
class SqliteFixtures:
    """SQLite's concrete `ConformanceFixtures` — structurally satisfies the Protocol.

    `SELECT FROM nope` is a genuine syntax error in SQLite (unlike Postgres, where it is only a
    missing-relation error), so it serves as the `parse_error` case. The `slow_query` recursive
    CTE mirrors DuckDB's: SQLite counts row by row and honours `interrupt()` between steps.
    """

    one_row_one_column: str = "SELECT 1 AS n"
    empty_result: str = "SELECT 1 AS n WHERE 1=0"
    three_rows: str = "SELECT 1 AS n UNION ALL SELECT 2 UNION ALL SELECT 3"
    null_value: str = "SELECT NULL AS x"
    duplicate_column_names: str = "SELECT 1 AS x, 2 AS x"
    references_missing_table: str = "SELECT * FROM does_not_exist_xyz"
    parse_error: str = "SELECT FROM nope"
    slow_query: str = (
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 100000000) "
        "SELECT count(*) AS n FROM t"
    )
    reports_types: bool = False


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
    reports_types: bool = True


@dataclass(frozen=True)
class DatabricksFixtures:
    """Databricks SQL Warehouse's concrete `ConformanceFixtures` — Spark SQL idiom.

    `range(...)` is Spark's table-valued generator (column `id`); the `slow_query` cross-joins
    two ranges under a row-forcing predicate so the count cannot be shortcut from cardinalities,
    overrunning a sub-second budget while remaining interruptible.
    """

    one_row_one_column: str = "SELECT 1 AS n"
    empty_result: str = "SELECT 1 AS n WHERE false"
    three_rows: str = "SELECT n FROM (VALUES (1), (2), (3)) AS t(n)"
    null_value: str = "SELECT NULL AS x"
    duplicate_column_names: str = "SELECT 1 AS x, 2 AS x"
    references_missing_table: str = "SELECT * FROM does_not_exist_xyz"
    parse_error: str = "SLECT 1"
    slow_query: str = "SELECT count(*) AS n FROM range(0, 50000) a CROSS JOIN range(0, 50000) b WHERE a.id + b.id > -1"
    reports_types: bool = True


@dataclass(frozen=True)
class SnowflakeFixtures:
    """Snowflake's concrete `ConformanceFixtures` — Snowflake SQL idiom.

    Unlike Databricks, Snowflake's driver reports precise column types (including
    `is_nullable`) directly, so `reports_types` is `True` with no probe needed.
    """

    one_row_one_column: str = "SELECT 1 AS n"
    empty_result: str = "SELECT 1 AS n WHERE 1=0"
    three_rows: str = "SELECT n FROM (VALUES (1), (2), (3)) AS t(n)"
    null_value: str = "SELECT NULL AS x"
    duplicate_column_names: str = "SELECT 1 AS x, 2 AS x"
    references_missing_table: str = "SELECT * FROM does_not_exist_xyz"
    parse_error: str = "SLECT 1"
    # NOTE: tuned for an XS warehouse; duration needs live confirmation and may need adjustment.
    slow_query: str = "SELECT MAX(SEQ8()) AS n FROM TABLE(GENERATOR(ROWCOUNT => 1000000000))"
    reports_types: bool = True


@dataclass(frozen=True)
class UnderTest:
    """One adapter-under-test: its live `PlatformAdapter` + the SQL it uses."""

    adapter: PlatformAdapter
    fixtures: ConformanceFixtures
    dialect: Dialect


def _duckdb_under_test() -> UnderTest:
    return UnderTest(adapter=DuckDBAdapter(), fixtures=DuckDBFixtures(), dialect="duckdb")


def _sqlite_under_test() -> UnderTest:
    return UnderTest(adapter=SqliteAdapter(), fixtures=SqliteFixtures(), dialect="sqlite")


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


def connect_postgres() -> PlatformAdapter:
    """Connect a `PostgresAdapter` to the configured test database.

    Fail-loud: a missing extra or unreachable database raises rather than skipping.
    """
    from evaldata.platforms.postgres import PostgresAdapter

    return PostgresAdapter(_postgres_dsn())


def _postgres_under_test() -> UnderTest:
    return UnderTest(adapter=connect_postgres(), fixtures=PostgresFixtures(), dialect="postgres")


def connect_databricks() -> PlatformAdapter:
    """Connect a `DatabricksAdapter` to the configured workspace.

    Reads `DATABRICKS_SERVER_HOSTNAME` and `DATABRICKS_HTTP_PATH`; credentials resolve from
    the ambient environment via the Databricks SDK. Fail-loud: a missing extra, unset
    connection details, or an unreachable workspace raises rather than skipping.
    """
    from evaldata.platforms.databricks import DatabricksAdapter

    return DatabricksAdapter(
        server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
    )


def _databricks_under_test() -> UnderTest:
    return UnderTest(adapter=connect_databricks(), fixtures=DatabricksFixtures(), dialect="databricks")


def connect_snowflake() -> PlatformAdapter:
    """Connect a `SnowflakeAdapter` to the configured account.

    Reads `SNOWFLAKE_ACCOUNT` (required); `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`,
    `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_ROLE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`,
    `SNOWFLAKE_AUTHENTICATOR`, `SNOWFLAKE_TOKEN`, `SNOWFLAKE_PRIVATE_KEY_FILE`, and
    `SNOWFLAKE_PRIVATE_KEY_FILE_PWD` are optional. Fail-loud: a missing extra, unset account, or
    an unreachable account raises rather than skipping.
    """
    from evaldata.platforms.snowflake import SnowflakeAdapter

    return SnowflakeAdapter(
        account=os.environ["SNOWFLAKE_ACCOUNT"],
        user=os.environ.get("SNOWFLAKE_USER"),
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        warehouse=os.environ.get("SNOWFLAKE_WAREHOUSE"),
        role=os.environ.get("SNOWFLAKE_ROLE"),
        database=os.environ.get("SNOWFLAKE_DATABASE"),
        schema=os.environ.get("SNOWFLAKE_SCHEMA"),
        authenticator=os.environ.get("SNOWFLAKE_AUTHENTICATOR"),
        token=os.environ.get("SNOWFLAKE_TOKEN"),
        private_key_file=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"),
        private_key_file_pwd=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"),
        workload_identity_provider=os.environ.get("SNOWFLAKE_WORKLOAD_IDENTITY_PROVIDER"),
    )


def _snowflake_under_test() -> UnderTest:
    return UnderTest(adapter=connect_snowflake(), fixtures=SnowflakeFixtures(), dialect="snowflake")


# Function-scoped: each test gets a fresh adapter.
@pytest.fixture(
    params=[
        pytest.param(_duckdb_under_test, id="duckdb", marks=pytest.mark.unit),
        pytest.param(_sqlite_under_test, id="sqlite", marks=pytest.mark.unit),
        pytest.param(_postgres_under_test, id="postgres", marks=pytest.mark.e2e),
        pytest.param(_databricks_under_test, id="databricks", marks=[pytest.mark.e2e, pytest.mark.cloud]),
        pytest.param(
            _snowflake_under_test, id="snowflake", marks=[pytest.mark.e2e, pytest.mark.cloud, pytest.mark.snowflake]
        ),
    ],
)
def under_test(request: pytest.FixtureRequest) -> UnderTest:
    """Return one (adapter, fixtures) pair; parametrised across all registered adapters."""
    factory = request.param
    return factory()
