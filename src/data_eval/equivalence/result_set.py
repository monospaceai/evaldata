"""Engine-input result-set types: `TypedResultSet` (with schema) and `UntypedResultSet` (rows only)."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from data_eval.types import Schema


class UntypedResultSet(BaseModel):
    """A result set without column-type information; type comparison is unavailable."""

    model_config = ConfigDict(extra="forbid")

    rows: list[dict[str, Any]]


class TypedResultSet(BaseModel):
    """A result set carrying column types; enables semantic type comparison via SQLGlot."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    rows: list[dict[str, Any]]
    schema_: Schema = Field(alias="schema")
