"""Core public Pydantic types for data-eval."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

PlatformKind = Literal["snowflake", "bigquery", "databricks", "postgres", "duckdb"]

SQLDialect = Literal[
    "snowflake",
    "bigquery",
    "databricks",
    "spark",
    "postgres",
    "redshift",
    "duckdb",
]


class PlatformRef(BaseModel):
    """Serializable reference to a configured data platform connection."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    kind: PlatformKind
    dialect: SQLDialect | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SnapshotRef(BaseModel):
    """Reference to a pinned platform-state snapshot for reproducible evaluation."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["timestamp", "version", "snapshot_id"]
    value: Annotated[str, Field(min_length=1)]


class ExpectedResultSet(BaseModel):
    """Expected outcome specified as concrete result-set rows."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    kind: Literal["result_set"] = "result_set"
    rows: list[dict[str, Any]]
    schema_: dict[str, str] | None = Field(default=None, alias="schema")


class ExpectedSQL(BaseModel):
    """Expected outcome specified as gold SQL; executed at eval time to yield the expected rows."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["sql"] = "sql"
    sql: Annotated[str, Field(min_length=1)]


class RowCountExpectation(BaseModel):
    """The result set must contain exactly this many rows."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["row_count"] = "row_count"
    exact: Annotated[int, Field(ge=0)]


class ColumnPresenceExpectation(BaseModel):
    """The result set must contain at least these columns."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["column_presence"] = "column_presence"
    columns: Annotated[list[str], Field(min_length=1)]


class ColumnTypeExpectation(BaseModel):
    """A named column must have the given SQL type."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["column_type"] = "column_type"
    column: Annotated[str, Field(min_length=1)]
    expected_type: Annotated[str, Field(min_length=1)]


class NonNullExpectation(BaseModel):
    """A named column must contain no NULL values."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["non_null"] = "non_null"
    column: Annotated[str, Field(min_length=1)]


class UniqueExpectation(BaseModel):
    """A named column's values must be distinct."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["unique"] = "unique"
    column: Annotated[str, Field(min_length=1)]


Expectation = Annotated[
    RowCountExpectation | ColumnPresenceExpectation | ColumnTypeExpectation | NonNullExpectation | UniqueExpectation,
    Field(discriminator="kind"),
]


class ExpectationSuite(BaseModel):
    """Expected outcome specified as a suite of expectations the result set must satisfy."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["expectation_suite"] = "expectation_suite"
    expectations: Annotated[list[Expectation], Field(min_length=1)]


Expected = Annotated[
    ExpectedResultSet | ExpectedSQL | ExpectationSuite,
    Field(discriminator="kind"),
]


class ComparisonConfig(BaseModel):
    """Rules for deciding whether two result sets are equivalent."""

    model_config = ConfigDict(extra="forbid")

    row_order: Literal["ignore", "strict"] = "ignore"
    column_order: Literal["ignore", "strict"] = "ignore"
    type_equality: Literal["ignore", "strict"] = "ignore"
    null_equality: Literal["equal", "distinct"] = "equal"
    float_tolerance: Annotated[float, Field(ge=0.0)] = 1e-9
