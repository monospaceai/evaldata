"""Platform adapters: per-platform integrations that execute SQL against a data platform."""

import importlib
from typing import TYPE_CHECKING, Any

from evaldata.platforms.base import PlatformAdapter
from evaldata.platforms.duckdb import DuckDBAdapter
from evaldata.platforms.registry import (
    bigquery_platform,
    databricks_platform,
    duckdb_platform,
    postgres_platform,
    resolve,
    snowflake_platform,
    sqlite_platform,
)
from evaldata.platforms.sqlite import SqliteAdapter

if TYPE_CHECKING:
    from evaldata.platforms.bigquery import BigQueryAdapter
    from evaldata.platforms.databricks import DatabricksAdapter
    from evaldata.platforms.postgres import PostgresAdapter
    from evaldata.platforms.snowflake import SnowflakeAdapter

__all__ = [
    "BigQueryAdapter",
    "DatabricksAdapter",
    "DuckDBAdapter",
    "PlatformAdapter",
    "PostgresAdapter",
    "SnowflakeAdapter",
    "SqliteAdapter",
    "bigquery_platform",
    "databricks_platform",
    "duckdb_platform",
    "postgres_platform",
    "resolve",
    "snowflake_platform",
    "sqlite_platform",
]

_LAZY_ADAPTERS = {
    "PostgresAdapter": ("evaldata.platforms.postgres", "postgres"),
    "DatabricksAdapter": ("evaldata.platforms.databricks", "databricks"),
    "SnowflakeAdapter": ("evaldata.platforms.snowflake", "snowflake"),
    "BigQueryAdapter": ("evaldata.platforms.bigquery", "bigquery"),
}


def __getattr__(name: str) -> Any:
    lazy = _LAZY_ADAPTERS.get(name)
    if lazy is not None:
        module_path, extra = lazy
        try:
            module = importlib.import_module(module_path)
        except ImportError as e:
            msg = f"{name} requires the {extra!r} extra: install evaldata[{extra}]"
            raise ImportError(msg) from e
        return getattr(module, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY_ADAPTERS])
