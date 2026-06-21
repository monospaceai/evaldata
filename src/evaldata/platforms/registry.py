"""Platform-reference builders and `PlatformRef` -> live `PlatformAdapter` resolution."""

from typing import assert_never

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.duckdb import DuckDBAdapter
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


def _build_duckdb(ref: PlatformRef) -> PlatformAdapter:
    return DuckDBAdapter(database=str(ref.config.get("path", ":memory:")))


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
        case "postgres":
            return _build_postgres(ref)
        case "databricks":
            return _build_databricks(ref)
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
