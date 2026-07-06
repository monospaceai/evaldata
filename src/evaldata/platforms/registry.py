"""Platform-reference builders and `PlatformRef` -> live `PlatformAdapter` resolution."""

import os
from typing import assert_never

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.duckdb import DuckDBAdapter
from evaldata.platforms.sqlite import SqliteAdapter
from evaldata.types import PlatformKind, PlatformRef


def duckdb_platform(name: str, path: str = ":memory:") -> PlatformRef:
    """Build a `PlatformRef` for an in-process DuckDB database.

    Args:
        name: A unique name identifying this platform connection.
        path: The DuckDB database path. Defaults to `:memory:` (in-process).

    Returns:
        A serializable `PlatformRef` for the DuckDB database. Building the ref needs no
        driver.
    """
    return PlatformRef(name=name, kind="duckdb", config={"path": path})


def sqlite_platform(name: str, path: str = ":memory:") -> PlatformRef:
    """Build a `PlatformRef` for an in-process SQLite database.

    Args:
        name: A unique name identifying this platform connection.
        path: The SQLite database path. Defaults to `:memory:` (in-process).

    Returns:
        A serializable `PlatformRef` for the SQLite database. Building the ref needs no driver
        (SQLite ships with the standard library).
    """
    return PlatformRef(name=name, kind="sqlite", config={"path": path})


def postgres_platform(name: str, conninfo: str = "") -> PlatformRef:
    """Build a `PlatformRef` for a PostgreSQL database.

    Args:
        name: A unique name identifying this platform connection.
        conninfo: A libpq connection string (keyword/value or `postgresql://` URI). Empty
            uses libpq defaults / `PG*` env vars.

    Returns:
        A serializable `PlatformRef` for the PostgreSQL database. Building the ref needs no
        driver.
    """
    return PlatformRef(name=name, kind="postgres", config={"conninfo": conninfo})


def databricks_platform(
    name: str,
    *,
    server_hostname: str,
    http_path: str,
    catalog: str | None = None,
    schema: str | None = None,
) -> PlatformRef:
    """Build a `PlatformRef` for a Databricks SQL Warehouse.

    Holds only non-secret connection details; credentials are not included here.

    Args:
        name: A unique name identifying this platform connection.
        server_hostname: The workspace hostname (no scheme), e.g. `dbc-xxxx.cloud.databricks.com`.
        http_path: The SQL Warehouse HTTP path, e.g. `/sql/1.0/warehouses/<id>`.
        catalog: The default catalog, or `None` to leave the session default.
        schema: The default schema, or `None` to leave the session default.

    Returns:
        A serializable `PlatformRef` for the Databricks warehouse. Building the ref needs no
        driver.
    """
    config: dict[str, str] = {"server_hostname": server_hostname, "http_path": http_path}
    if catalog is not None:
        config["catalog"] = catalog
    if schema is not None:
        config["schema"] = schema
    return PlatformRef(name=name, kind="databricks", dialect="databricks", config=config)


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
) -> PlatformRef:
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
        authenticator: The authenticator to use (e.g. `"externalbrowser"`, `"oauth"`),
            or `None` for the connector's default.

    Returns:
        A serializable `PlatformRef` for the Snowflake account. Building the ref needs no
        driver.
    """
    fields = {
        "user": user,
        "warehouse": warehouse,
        "role": role,
        "database": database,
        "schema": schema,
        "authenticator": authenticator,
    }
    config: dict[str, str] = {"account": account}
    config.update({k: v for k, v in fields.items() if v is not None})
    return PlatformRef(name=name, kind="snowflake", dialect="snowflake", config=config)


def _build_duckdb(ref: PlatformRef) -> PlatformAdapter:
    return DuckDBAdapter(database=str(ref.config.get("path", ":memory:")))


def _build_sqlite(ref: PlatformRef) -> PlatformAdapter:
    return SqliteAdapter(database=str(ref.config.get("path", ":memory:")))


def _build_postgres(ref: PlatformRef) -> PlatformAdapter:
    try:
        from evaldata.platforms.postgres import PostgresAdapter
    except ImportError as e:
        msg = "PostgresAdapter requires the 'postgres' extra; install it with `uv sync --extra postgres`"
        raise RuntimeError(msg) from e
    return PostgresAdapter(conninfo=str(ref.config.get("conninfo", "")))


def _build_databricks(ref: PlatformRef) -> PlatformAdapter:
    try:
        from evaldata.platforms.databricks import DatabricksAdapter
    except ImportError as e:
        msg = "DatabricksAdapter requires the 'databricks' extra; install it with `uv sync --extra databricks`"
        raise RuntimeError(msg) from e
    catalog = ref.config.get("catalog")
    schema = ref.config.get("schema")
    return DatabricksAdapter(
        server_hostname=str(ref.config["server_hostname"]),
        http_path=str(ref.config["http_path"]),
        catalog=str(catalog) if catalog is not None else None,
        schema=str(schema) if schema is not None else None,
    )


def _build_snowflake(ref: PlatformRef) -> PlatformAdapter:
    try:
        from evaldata.platforms.snowflake import SnowflakeAdapter
    except ImportError as e:
        msg = "SnowflakeAdapter requires the 'snowflake' extra; install it with `uv sync --extra snowflake`"
        raise RuntimeError(msg) from e
    config = ref.config
    return SnowflakeAdapter(
        account=str(config["account"]),
        user=str(config["user"]) if "user" in config else None,
        password=os.environ.get("SNOWFLAKE_PASSWORD"),
        warehouse=str(config["warehouse"]) if "warehouse" in config else None,
        role=str(config["role"]) if "role" in config else None,
        database=str(config["database"]) if "database" in config else None,
        schema=str(config["schema"]) if "schema" in config else None,
        authenticator=str(config["authenticator"]) if "authenticator" in config else None,
        token=os.environ.get("SNOWFLAKE_TOKEN"),
        private_key_file=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE"),
        private_key_file_pwd=os.environ.get("SNOWFLAKE_PRIVATE_KEY_FILE_PWD"),
        workload_identity_provider=str(config["workload_identity_provider"])
        if "workload_identity_provider" in config
        else None,
    )


def _build(ref: PlatformRef) -> PlatformAdapter:
    """Build a live adapter for `ref` by exhaustive dispatch over its kind.

    Args:
        ref: The platform reference to build an adapter for.

    Returns:
        A live `PlatformAdapter` for the reference's platform kind.
    """
    kind: PlatformKind = ref.kind
    match kind:
        case "duckdb":
            return _build_duckdb(ref)
        case "sqlite":
            return _build_sqlite(ref)
        case "postgres":
            return _build_postgres(ref)
        case "databricks":
            return _build_databricks(ref)
        case "snowflake":
            return _build_snowflake(ref)
        case _ as unreachable:  # pragma: no cover - exhaustiveness guard
            assert_never(unreachable)


_ADAPTERS: dict[str, tuple[PlatformRef, PlatformAdapter]] = {}


def resolve(ref: PlatformRef) -> PlatformAdapter:
    """Return a live adapter for `ref`, building and caching one on first use.

    Reuses the cached adapter on subsequent calls for the same `ref.name`. An unsupported
    `kind` is unrepresentable — `PlatformRef` validation rejects it before resolution.

    Args:
        ref: The platform reference to resolve.

    Returns:
        The live `PlatformAdapter` for `ref`, cached and reused across calls.

    Raises:
        ValueError: If `ref.name` is already bound to a different configuration.
    """
    cached = _ADAPTERS.get(ref.name)
    if cached is not None:
        cached_ref, adapter = cached
        if cached_ref != ref:
            msg = (
                f"platform name {ref.name!r} is already bound to a different configuration; "
                "platform names must uniquely identify a connection"
            )
            raise ValueError(msg)
        return adapter

    adapter = _build(ref)
    _ADAPTERS[ref.name] = (ref, adapter)
    return adapter


def close_all() -> None:
    """Close every cached adapter and clear the cache (idempotent; no-op when empty)."""
    for _ref, adapter in _ADAPTERS.values():
        adapter.close()
    _ADAPTERS.clear()
