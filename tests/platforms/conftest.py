"""Conformance plumbing: `ConformanceFixtures` Protocol, per-adapter fixture types, and the
parametrised `under_test` fixture, plus the shared `ADAPTER_SPECS` registry that every
scorer-conformance suite's `engine` fixture derives its adapter list from.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pytest
import sqlglot
from sqlglot import exp
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.duckdb import DuckDBAdapter
from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.scorers.sql import Dialect
from evaldata.types import PlatformKind


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
    duplicate_column_names: str  # produces two output columns named "x" (renamed by some engines)
    references_missing_table: str  # references a non-existent table
    parse_error: str  # is syntactically invalid
    slow_query: str  # runs long enough to overrun a sub-second budget, and is interruptible
    reports_types: bool  # whether the driver reports result-column types (False for SQLite)
    renames_duplicate_columns: bool  # whether the engine disambiguates duplicate output names itself (BigQuery)


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
    renames_duplicate_columns: bool = False


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
    renames_duplicate_columns: bool = False


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
    renames_duplicate_columns: bool = False


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
    renames_duplicate_columns: bool = False


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
    renames_duplicate_columns: bool = False


@dataclass(frozen=True)
class BigQueryFixtures:
    """BigQuery's concrete `ConformanceFixtures` — GoogleSQL idiom.

    BigQuery's driver reports precise result-column types directly, so `reports_types` is `True`
    with no probe needed. `UNNEST` supplies the row-list form GoogleSQL lacks as `VALUES ... AS
    t(n)`; the `slow_query` hashes a generated array, which scans ~0 bytes yet runs long enough to
    overrun a sub-second budget.
    """

    one_row_one_column: str = "SELECT 1 AS n"
    # A WHERE needs a FROM in BigQuery, so the empty result draws from a single-element UNNEST.
    empty_result: str = "SELECT 1 AS n FROM UNNEST([1]) WHERE 1=0"
    three_rows: str = "SELECT n FROM UNNEST([1, 2, 3]) AS n"
    null_value: str = "SELECT NULL AS x"
    # BigQuery disambiguates the second `x` to `x_1` rather than returning a true collision.
    duplicate_column_names: str = "SELECT 1 AS x, 2 AS x"
    references_missing_table: str = "SELECT * FROM does_not_exist_xyz"
    parse_error: str = "SLECT 1"
    slow_query: str = (
        "SELECT SUM(LENGTH(TO_HEX(SHA256(CAST(n AS STRING))))) AS n FROM UNNEST(GENERATE_ARRAY(1, 1000000)) AS n"
    )
    reports_types: bool = True
    renames_duplicate_columns: bool = True


@dataclass(frozen=True)
class UnderTest:
    """One adapter-under-test: its live `PlatformAdapter` + the SQL it uses."""

    adapter: PlatformAdapter
    fixtures: ConformanceFixtures
    dialect: Dialect


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


def connect_bigquery() -> PlatformAdapter:
    """Connect a `BigQueryAdapter` to the configured project.

    Reads `BIGQUERY_PROJECT` (required); `BIGQUERY_DATASET` and `BIGQUERY_LOCATION` are optional.
    Credentials resolve from the ambient environment via Application Default Credentials.
    Fail-loud: a missing extra, unset project, or an unreachable project raises rather than
    skipping.
    """
    from evaldata.platforms.bigquery import BigQueryAdapter

    return BigQueryAdapter(
        project=os.environ["BIGQUERY_PROJECT"],
        dataset=os.environ.get("BIGQUERY_DATASET"),
        location=os.environ.get("BIGQUERY_LOCATION"),
    )


def connect_sqlite() -> PlatformAdapter:
    """Connect a `SqliteAdapter` to a fresh in-memory database."""
    return SqliteAdapter()


@dataclass(frozen=True)
class AdapterSpec:
    """One registered platform adapter: its id, pytest marks, connector, and conformance fixtures.

    The single registry `under_test` and every scorer-conformance suite's `engine` fixture derive
    their parametrised adapter list from, so adding or removing a supported adapter is a one-line
    change here rather than an edit repeated (and liable to drift) across several test files.
    """

    id: PlatformKind
    marks: pytest.MarkDecorator | list[pytest.MarkDecorator]
    connect: Callable[[], PlatformAdapter]
    fixtures: Callable[[], ConformanceFixtures]


ADAPTER_SPECS: list[AdapterSpec] = [
    AdapterSpec(id="duckdb", marks=pytest.mark.unit, connect=DuckDBAdapter, fixtures=DuckDBFixtures),
    AdapterSpec(id="sqlite", marks=pytest.mark.unit, connect=connect_sqlite, fixtures=SqliteFixtures),
    AdapterSpec(id="postgres", marks=pytest.mark.e2e, connect=connect_postgres, fixtures=PostgresFixtures),
    AdapterSpec(
        id="databricks",
        marks=[pytest.mark.e2e, pytest.mark.cloud],
        connect=connect_databricks,
        fixtures=DatabricksFixtures,
    ),
    AdapterSpec(
        id="snowflake",
        marks=[pytest.mark.e2e, pytest.mark.cloud, pytest.mark.snowflake],
        connect=connect_snowflake,
        fixtures=SnowflakeFixtures,
    ),
    AdapterSpec(
        id="bigquery",
        marks=[pytest.mark.e2e, pytest.mark.cloud, pytest.mark.bigquery],
        connect=connect_bigquery,
        fixtures=BigQueryFixtures,
    ),
]

# The id set `ADAPTER_SPECS` covers; a completeness guard asserts this equals every `PlatformKind`.
ADAPTER_IDS: frozenset[str] = frozenset(spec.id for spec in ADAPTER_SPECS)


def engine_params() -> list[Any]:
    """The `pytest.param` list for an `(adapter, dialect)` `engine` fixture over every registered adapter.

    Every scorer-conformance suite (`test_conformance_equivalence.py`, `test_conformance_pushdown.py`)
    builds its `engine` fixture from this, so their adapter coverage can never silently drift from
    `under_test`'s or from each other's.
    """

    def _factory(spec: AdapterSpec) -> Callable[[], tuple[PlatformAdapter, PlatformKind]]:
        def make() -> tuple[PlatformAdapter, PlatformKind]:
            return spec.connect(), spec.id

        return make

    return [pytest.param(_factory(spec), id=spec.id, marks=spec.marks) for spec in ADAPTER_SPECS]


# Function-scoped: each test gets a fresh adapter.
@pytest.fixture(params=[pytest.param(spec, id=spec.id, marks=spec.marks) for spec in ADAPTER_SPECS])
def under_test(request: pytest.FixtureRequest) -> UnderTest:
    """Return one (adapter, fixtures) pair; parametrised across all registered adapters."""
    spec: AdapterSpec = request.param
    return UnderTest(adapter=spec.connect(), fixtures=spec.fixtures(), dialect=spec.id)
