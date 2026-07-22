"""Platform-reference builders and `PlatformRef` -> live `PlatformAdapter` resolution."""

import os
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import assert_never
from urllib.parse import quote

import duckdb

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.duckdb import DuckDBAdapter
from evaldata.platforms.pool import ConnectionPool
from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.types import (
    BigQueryConfig,
    BigQueryPlatformRef,
    DatabricksConfig,
    DatabricksPlatformRef,
    DuckDBConfig,
    DuckDBPlatformRef,
    PlatformKind,
    PlatformRef,
    PoolPolicy,
    PostgreSQLConfig,
    PostgreSQLPlatformRef,
    SnowflakeConfig,
    SnowflakePlatformRef,
    SQLiteConfig,
    SQLitePlatformRef,
)


def duckdb_platform(name: str, path: str = ":memory:", *, pool: PoolPolicy | None = None) -> DuckDBPlatformRef:
    """Build a `PlatformRef` for an in-process DuckDB database.

    Args:
        name: A unique name identifying this platform connection.
        path: The DuckDB database path. Defaults to `:memory:` (in-process).
        pool: Optional connection lifecycle policy.

    Returns:
        A serializable `PlatformRef` for the DuckDB database. Building the ref needs no
        driver.
    """
    return DuckDBPlatformRef(name=name, config=DuckDBConfig(path=path), pool=pool)


def sqlite_platform(name: str, path: str = ":memory:", *, pool: PoolPolicy | None = None) -> SQLitePlatformRef:
    """Build a `PlatformRef` for an in-process SQLite database.

    Args:
        name: A unique name identifying this platform connection.
        path: The SQLite database path. Defaults to `:memory:` (in-process).
        pool: Optional connection lifecycle policy.

    Returns:
        A serializable `PlatformRef` for the SQLite database. Building the ref needs no driver
        (SQLite ships with the standard library).
    """
    return SQLitePlatformRef(name=name, config=SQLiteConfig(path=path), pool=pool)


def postgres_platform(name: str, conninfo: str = "", *, pool: PoolPolicy | None = None) -> PostgreSQLPlatformRef:
    """Build a `PlatformRef` for a PostgreSQL database.

    Args:
        name: A unique name identifying this platform connection.
        conninfo: A libpq connection string (keyword/value or `postgresql://` URI). Empty
            uses libpq defaults / `PG*` env vars.
        pool: Optional connection lifecycle policy.

    Returns:
        A serializable `PlatformRef` for the PostgreSQL database. Building the ref needs no
        driver.
    """
    return PostgreSQLPlatformRef(name=name, config=PostgreSQLConfig(conninfo=conninfo), pool=pool)


def databricks_platform(
    name: str,
    *,
    server_hostname: str,
    http_path: str,
    catalog: str | None = None,
    schema: str | None = None,
    pool: PoolPolicy | None = None,
) -> DatabricksPlatformRef:
    """Build a `PlatformRef` for a Databricks SQL Warehouse.

    Holds only non-secret connection details; credentials are not included here.

    Args:
        name: A unique name identifying this platform connection.
        server_hostname: The workspace hostname (no scheme), e.g. `dbc-xxxx.cloud.databricks.com`.
        http_path: The SQL Warehouse HTTP path, e.g. `/sql/1.0/warehouses/<id>`.
        catalog: The default catalog, or `None` to leave the session default.
        schema: The default schema, or `None` to leave the session default.
        pool: Optional connection lifecycle policy.

    Returns:
        A serializable `PlatformRef` for the Databricks warehouse. Building the ref needs no
        driver.
    """
    config = DatabricksConfig(
        server_hostname=server_hostname,
        http_path=http_path,
        catalog=catalog,
        schema=schema,
    )
    return DatabricksPlatformRef(name=name, dialect="databricks", config=config, pool=pool)


def snowflake_platform(
    name: str,
    *,
    account: str,
    user: str | None = None,
    warehouse: str | None = None,
    role: str | None = None,
    database: str | None = None,
    schema: str | None = None,
    authenticator: str | None = None,
    workload_identity_provider: str | None = None,
    pool: PoolPolicy | None = None,
) -> SnowflakePlatformRef:
    """Build a `PlatformRef` for a Snowflake account.

    Holds only non-secret connection details; credentials are not included here.

    Args:
        name: A unique name identifying this platform connection.
        account: The Snowflake account identifier.
        user: The Snowflake user name, or `None` to rely on `authenticator`.
        warehouse: The default warehouse, or `None` to leave the session default.
        role: The default role, or `None` to leave the session default.
        database: The default database, or `None` to leave the session default.
        schema: The default schema, or `None` to leave the session default.
        authenticator: The authenticator to use (e.g. `"externalbrowser"`, `"oauth"`,
            `"workload_identity"`), or `None` for the connector's default.
        workload_identity_provider: The workload identity provider (e.g. `"OIDC"`) when
            `authenticator` is `"workload_identity"`, or `None` otherwise.
        pool: Optional connection lifecycle policy.

    Returns:
        A serializable `PlatformRef` for the Snowflake account. Building the ref needs no
        driver.
    """
    config = SnowflakeConfig(
        account=account,
        user=user,
        warehouse=warehouse,
        role=role,
        database=database,
        schema=schema,
        authenticator=authenticator,
        workload_identity_provider=workload_identity_provider,
    )
    return SnowflakePlatformRef(name=name, dialect="snowflake", config=config, pool=pool)


def bigquery_platform(
    name: str,
    *,
    project: str,
    dataset: str | None = None,
    location: str | None = None,
    pool: PoolPolicy | None = None,
) -> BigQueryPlatformRef:
    """Build a `PlatformRef` for a BigQuery project.

    Holds only non-secret connection details; credentials are not included here.

    Args:
        name: A unique name identifying this platform connection.
        project: The Google Cloud project to run jobs and bill against.
        dataset: The default dataset for unqualified table names, or `None` to leave none.
        location: The location to run jobs in (e.g. `"US"`, `"EU"`), or `None` for the client
            default.
        pool: Optional connection lifecycle policy.

    Returns:
        A serializable `PlatformRef` for the BigQuery project. Building the ref needs no driver.
    """
    config = BigQueryConfig(project=project, dataset=dataset, location=location)
    return BigQueryPlatformRef(name=name, dialect="bigquery", config=config, pool=pool)


def _build_sqlite(ref: SQLitePlatformRef) -> PlatformAdapter:
    path = ref.config.path
    # Keep utility and member connections on the same in-memory database.
    database = f"file:evaldata-{quote(ref.name, safe='')}?mode=memory&cache=shared" if path == ":memory:" else path
    return SqliteAdapter(database=database)


def _build_postgres(ref: PostgreSQLPlatformRef) -> PlatformAdapter:
    try:
        from evaldata.platforms.postgres import PostgresAdapter
    except ImportError as e:
        msg = "PostgresAdapter requires the 'postgres' extra; install it with `uv sync --extra postgres`"
        raise RuntimeError(msg) from e
    return PostgresAdapter(conninfo=ref.config.conninfo)


def _build_databricks(ref: DatabricksPlatformRef) -> PlatformAdapter:
    try:
        from evaldata.platforms.databricks import DatabricksAdapter
    except ImportError as e:
        msg = "DatabricksAdapter requires the 'databricks' extra; install it with `uv sync --extra databricks`"
        raise RuntimeError(msg) from e
    return DatabricksAdapter(
        server_hostname=ref.config.server_hostname,
        http_path=ref.config.http_path,
        catalog=ref.config.catalog,
        schema=ref.config.schema_,
    )


def _build_snowflake(ref: SnowflakePlatformRef) -> PlatformAdapter:
    try:
        from evaldata.platforms.snowflake import SnowflakeAdapter
    except ImportError as e:
        msg = "SnowflakeAdapter requires the 'snowflake' extra; install it with `uv sync --extra snowflake`"
        raise RuntimeError(msg) from e
    config = ref.config
    return SnowflakeAdapter(
        account=config.account,
        user=config.user,
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        warehouse=config.warehouse,
        role=config.role,
        database=config.database,
        schema=config.schema_,
        authenticator=config.authenticator,
        token=os.environ.get("SNOWFLAKE_TOKEN"),
        private_key_file=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"),
        private_key_file_pwd=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"),
        workload_identity_provider=config.workload_identity_provider,
    )


def _build_bigquery(ref: BigQueryPlatformRef) -> PlatformAdapter:
    try:
        from evaldata.platforms.bigquery import BigQueryAdapter
    except ImportError as e:
        msg = "BigQueryAdapter requires the 'bigquery' extra; install it with `uv sync --extra bigquery`"
        raise RuntimeError(msg) from e
    config = ref.config
    return BigQueryAdapter(
        project=config.project,
        dataset=config.dataset,
        location=config.location,
    )


_MAX_SIZE: dict[PlatformKind, int] = {
    "duckdb": 8,
    "sqlite": 1,
    "postgres": 4,
    "databricks": 4,
    "snowflake": 4,
    "bigquery": 4,
}


def _pool_policy(ref: PlatformRef) -> PoolPolicy:
    """Resolve `ref`'s explicit policy or the platform's bounded lifecycle defaults.

    Returns:
        The effective policy for `ref`.
    """
    if ref.pool is not None:
        return ref.pool
    return PoolPolicy(max_size=_MAX_SIZE[ref.kind], pre_ping=ref.kind in {"postgres", "snowflake", "databricks"})


def _same_platform(ref: PlatformRef, other: PlatformRef) -> bool:
    """Return whether two references have identical connection and effective-pool semantics.

    Returns:
        Whether the references can safely share one cached pool.
    """
    bare = ref.model_copy(update={"pool": None})
    other_bare = other.model_copy(update={"pool": None})
    return bare == other_bare and _pool_policy(ref) == _pool_policy(other)


def _duckdb_pool(ref: DuckDBPlatformRef) -> ConnectionPool:
    """Build a DuckDB pool whose members and utility are cursors of one shared parent connection.

    Returns:
        A `ConnectionPool` owning the shared parent, closed when the pool closes.
    """
    parent = duckdb.connect(ref.config.path)

    def factory() -> PlatformAdapter:
        return DuckDBAdapter.from_connection(parent.cursor())

    return ConnectionPool(ref, factory, policy=_pool_policy(ref), parent=parent)


def _dedicated_pool(ref: PlatformRef, build: Callable[[], PlatformAdapter]) -> ConnectionPool:
    """Build a pool whose every member and utility is an independent adapter (its own connection).

    Returns:
        A `ConnectionPool` whose members each own a fresh connection built by `build`.
    """

    def factory() -> PlatformAdapter:
        return build()

    return ConnectionPool(ref, factory, policy=_pool_policy(ref))


def _build_pool(ref: PlatformRef) -> ConnectionPool:
    """Build a `ConnectionPool` for `ref`.

    Args:
        ref: The platform reference to build a pool for.

    Returns:
        A `ConnectionPool` matching the reference's per-engine session model.
    """
    if isinstance(ref, DuckDBPlatformRef):
        return _duckdb_pool(ref)
    if isinstance(ref, SQLitePlatformRef):
        return _dedicated_pool(ref, lambda: _build_sqlite(ref))
    if isinstance(ref, PostgreSQLPlatformRef):
        return _dedicated_pool(ref, lambda: _build_postgres(ref))
    if isinstance(ref, DatabricksPlatformRef):
        return _dedicated_pool(ref, lambda: _build_databricks(ref))
    if isinstance(ref, SnowflakePlatformRef):
        return _dedicated_pool(ref, lambda: _build_snowflake(ref))
    if isinstance(ref, BigQueryPlatformRef):
        return _dedicated_pool(ref, lambda: _build_bigquery(ref))
    assert_never(ref)  # pragma: no cover


_POOLS: dict[str, ConnectionPool] = {}
_POOLS_LOCK = threading.Lock()


def pool_for(ref: PlatformRef) -> ConnectionPool:
    """Return the connection pool for `ref.name`, building and caching one on first use.

    Reuses the cached pool on subsequent calls for the same `ref.name`.

    Args:
        ref: The platform reference to resolve a pool for.

    Returns:
        The `ConnectionPool` for `ref`, cached and reused across calls.

    Raises:
        ValueError: If `ref.name` is already bound to a different configuration.
    """
    with _POOLS_LOCK:
        cached = _POOLS.get(ref.name)
        if cached is not None:
            if not _same_platform(cached.ref, ref):
                msg = (
                    f"platform name {ref.name!r} is already bound to a different configuration; "
                    "platform names must uniquely identify a connection"
                )
                raise ValueError(msg)
            return cached
        pool = _build_pool(ref)
        _POOLS[ref.name] = pool
        return pool


def resolve(ref: PlatformRef) -> PlatformAdapter:
    """Return `ref`'s dedicated utility adapter, building its pool on first use.

    The utility adapter is for seeding, `doctor`, and direct execution; it is never a checkout
    member, so its use never contends with concurrent case execution. Concurrent execution goes
    through `acquired`.

    Args:
        ref: The platform reference to resolve.

    Returns:
        The utility `PlatformAdapter` for `ref`, cached and reused across calls.
    """
    return pool_for(ref).utility()


@contextmanager
def acquired(ref: PlatformRef) -> Iterator[PlatformAdapter]:
    """Check out one of `ref`'s pool members for the duration of the `with` block.

    Yields a member reserved exclusively for the caller, returned to the pool on exit even if
    the block raises. Concurrent callers each get a distinct member, blocking once the pool's
    per-engine size is reached.

    Args:
        ref: The platform reference whose pool to acquire a member from.

    Yields:
        A `PlatformAdapter` reserved for the caller until the block exits.
    """
    pool = pool_for(ref)
    member = pool.acquire()
    try:
        yield member
    finally:
        pool.release(member)


def close_all() -> None:
    """Close every cached pool and clear the cache (idempotent; no-op when empty)."""
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()
