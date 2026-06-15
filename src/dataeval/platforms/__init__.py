"""Platform adapters: per-platform integrations that execute SQL against a data platform."""

from typing import TYPE_CHECKING, Any

from dataeval.platforms.base import PlatformAdapter
from dataeval.platforms.duckdb import DuckDBAdapter
from dataeval.platforms.registry import duckdb_platform, postgres_platform, resolve

if TYPE_CHECKING:
    from dataeval.platforms.postgres import PostgresAdapter

__all__ = [
    "DuckDBAdapter",
    "PlatformAdapter",
    "PostgresAdapter",
    "duckdb_platform",
    "postgres_platform",
    "resolve",
]


def __getattr__(name: str) -> Any:
    if name == "PostgresAdapter":
        try:
            from dataeval.platforms.postgres import PostgresAdapter
        except ImportError as e:
            msg = "PostgresAdapter requires the 'postgres' extra: install dataeval[postgres]"
            raise ImportError(msg) from e
        return PostgresAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted([*globals(), "PostgresAdapter"])
