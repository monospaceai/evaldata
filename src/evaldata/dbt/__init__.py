"""dbt integration: load a dbt project's artifacts into evaldata types.

`DbtContext` reads a built dbt `target/` directory (manifest.json + optional catalog.json)
and exposes models, sources, and schema context. `load_dbt` converts them into eval cases;
`platform_from_profile` resolves the project's warehouse connection from a dbt profile.
"""

from evaldata.dbt.context import (
    Column,
    DbtContext,
    DbtTest,
    ModelRef,
    Relation,
    SchemaContext,
    SourceRef,
    TableSchema,
)
from evaldata.dbt.errors import DbtError
from evaldata.dbt.loader import Mode, load_dbt
from evaldata.dbt.profile import platform_from_profile

__all__ = [
    "Column",
    "DbtContext",
    "DbtError",
    "DbtTest",
    "Mode",
    "ModelRef",
    "Relation",
    "SchemaContext",
    "SourceRef",
    "TableSchema",
    "load_dbt",
    "platform_from_profile",
]
