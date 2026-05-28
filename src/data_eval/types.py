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


class Column(BaseModel):
    """A result-set column: its name and native SQL type string (compared semantically via SQLGlot)."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    type: Annotated[str, Field(min_length=1)]


Schema = list[Column]


class ExpectedResultSet(BaseModel):
    """Expected outcome specified as concrete result-set rows."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    kind: Literal["result_set"] = "result_set"
    rows: list[dict[str, Any]]
    schema_: Schema | None = Field(default=None, alias="schema")


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


class NotNullExpectation(BaseModel):
    """A named column must contain no NULL values."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["not_null"] = "not_null"
    column: Annotated[str, Field(min_length=1)]


class UniqueExpectation(BaseModel):
    """A named column's values must be distinct."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["unique"] = "unique"
    column: Annotated[str, Field(min_length=1)]


Expectation = Annotated[
    RowCountExpectation | ColumnPresenceExpectation | ColumnTypeExpectation | NotNullExpectation | UniqueExpectation,
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

    column_order: Literal["ignore", "strict"] = "ignore"
    null_equality: Literal["equal", "distinct"] = "equal"
    float_tolerance: Annotated[float, Field(ge=0.0)] = 1e-9


class CostBudget(BaseModel):
    """Per-eval-case ceiling on platform resource consumption."""

    model_config = ConfigDict(extra="forbid")

    max_seconds: Annotated[float, Field(gt=0)] | None = None


class EvalCase(BaseModel):
    """One AI-evaluation case: an input with an expected outcome and a platform to run against."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    input: Annotated[str, Field(min_length=1)]
    expected: Expected
    platform: PlatformRef
    snapshot: SnapshotRef | None = None
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    allow_data_egress: bool = False
    cost_budget: CostBudget | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SolverOutput(BaseModel):
    """The raw output produced by a Solver; scorers interpret and extract domain content."""

    model_config = ConfigDict(extra="forbid")

    output: Annotated[str, Field(min_length=1)]
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    latency_seconds: Annotated[float, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """The result of running SQL against a platform: returned rows plus execution measurements."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    rows: list[dict[str, Any]]
    schema_: Schema | None = Field(default=None, alias="schema")
    latency_seconds: Annotated[float, Field(ge=0)]
    error: Annotated[str, Field(min_length=1)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TypeMismatch(BaseModel):
    """A column whose actual type in the result set differs from the expected type."""

    model_config = ConfigDict(extra="forbid")

    column: Annotated[str, Field(min_length=1)]
    expected: Annotated[str, Field(min_length=1)]
    actual: Annotated[str, Field(min_length=1)]


class ColumnMismatch(BaseModel):
    """Per-column count of rows whose value in the actual result set differs from the expected value."""

    model_config = ConfigDict(extra="forbid")

    column: Annotated[str, Field(min_length=1)]
    unexpected_count: Annotated[int, Field(ge=0)]


class ResultSetDiff(BaseModel):
    """Structured difference between an actual result set and an expected result set."""

    model_config = ConfigDict(extra="forbid")

    expected_row_count: Annotated[int, Field(ge=0)]
    actual_row_count: Annotated[int, Field(ge=0)]
    missing_row_count: Annotated[int, Field(ge=0)] = 0
    extra_row_count: Annotated[int, Field(ge=0)] = 0
    missing_columns: list[str] = Field(default_factory=list)
    extra_columns: list[str] = Field(default_factory=list)
    type_mismatches: list[TypeMismatch] = Field(default_factory=list)
    column_mismatches: list[ColumnMismatch] = Field(default_factory=list)


class ScoreResult(BaseModel):
    """The outcome of running a Scorer against an EvalCase: pass/fail plus diagnostics."""

    model_config = ConfigDict(extra="forbid")

    scorer: Annotated[str, Field(min_length=1)]
    passed: bool
    diff: ResultSetDiff | None = None
    explanation: Annotated[str, Field(min_length=1)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
