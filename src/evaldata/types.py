"""Core public Pydantic types for evaldata."""

from collections import Counter
from collections.abc import Iterator
from typing import Annotated, Any, Literal, NewType, TypeAlias

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, RootModel, model_validator
from sqlglot import exp
from sqlglot.errors import SqlglotError

Sql = NewType("Sql", str)

PlatformKind = Literal["duckdb", "postgres", "databricks", "snowflake", "bigquery", "sqlite"]

SQLDialect = Literal[
    "snowflake",
    "bigquery",
    "databricks",
    "spark",
    "postgres",
    "redshift",
    "duckdb",
    "sqlite",
]

Verdict = Literal["pass", "fail", "inconclusive"]

Basis = Literal["proven", "observed", "judged"]

Score = Annotated[float, Field(ge=0.0, le=1.0)]


class PoolPolicy(BaseModel):
    """Lifecycle limits for a platform's persistent connection pool."""

    model_config = ConfigDict(extra="forbid")

    max_size: Annotated[int, Field(gt=0)]
    pre_ping: bool = False
    acquire_timeout_seconds: Annotated[float, Field(gt=0)] = 30.0
    cancel_grace_seconds: Annotated[float, Field(ge=0)] = 1.0
    max_quarantined: Annotated[int, Field(gt=0)] | None = None
    max_lifetime_seconds: Annotated[float, Field(gt=0)] | None = None
    max_idle_seconds: Annotated[float, Field(gt=0)] | None = None


class DuckDBConfig(BaseModel):
    """DuckDB connection settings."""

    model_config = ConfigDict(extra="forbid")

    path: str = ":memory:"


class SQLiteConfig(BaseModel):
    """SQLite connection settings."""

    model_config = ConfigDict(extra="forbid")

    path: str = ":memory:"


class PostgreSQLConfig(BaseModel):
    """PostgreSQL connection settings."""

    model_config = ConfigDict(extra="forbid")

    conninfo: str = ""


class DatabricksConfig(BaseModel):
    """Non-secret Databricks connection settings."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    server_hostname: Annotated[str, Field(min_length=1)]
    http_path: Annotated[str, Field(min_length=1)]
    catalog: str | None = None
    schema_: str | None = Field(default=None, alias="schema")


class SnowflakeConfig(BaseModel):
    """Non-secret Snowflake connection settings."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    account: Annotated[str, Field(min_length=1)]
    user: str | None = None
    warehouse: str | None = None
    role: str | None = None
    database: str | None = None
    schema_: str | None = Field(default=None, alias="schema")
    authenticator: str | None = None
    workload_identity_provider: str | None = None


class BigQueryConfig(BaseModel):
    """Non-secret BigQuery connection settings."""

    model_config = ConfigDict(extra="forbid")

    project: Annotated[str, Field(min_length=1)]
    dataset: str | None = None
    location: str | None = None


class _PlatformRefBase(BaseModel):
    """Fields shared by platform references."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    dialect: SQLDialect | None = None
    pool: PoolPolicy | None = Field(default=None, exclude_if=lambda value: value is None)


class DuckDBPlatformRef(_PlatformRefBase):
    """Reference to a DuckDB database."""

    kind: Literal["duckdb"] = "duckdb"
    config: DuckDBConfig = Field(default_factory=DuckDBConfig)


class SQLitePlatformRef(_PlatformRefBase):
    """Reference to a SQLite database."""

    kind: Literal["sqlite"] = "sqlite"
    config: SQLiteConfig = Field(default_factory=SQLiteConfig)


class PostgreSQLPlatformRef(_PlatformRefBase):
    """Reference to a PostgreSQL database."""

    kind: Literal["postgres"] = "postgres"
    config: PostgreSQLConfig = Field(default_factory=PostgreSQLConfig)


class DatabricksPlatformRef(_PlatformRefBase):
    """Reference to a Databricks SQL Warehouse."""

    kind: Literal["databricks"] = "databricks"
    config: DatabricksConfig


class SnowflakePlatformRef(_PlatformRefBase):
    """Reference to a Snowflake account."""

    kind: Literal["snowflake"] = "snowflake"
    config: SnowflakeConfig


class BigQueryPlatformRef(_PlatformRefBase):
    """Reference to a BigQuery project."""

    kind: Literal["bigquery"] = "bigquery"
    config: BigQueryConfig


PlatformRef: TypeAlias = Annotated[
    DuckDBPlatformRef
    | SQLitePlatformRef
    | PostgreSQLPlatformRef
    | DatabricksPlatformRef
    | SnowflakePlatformRef
    | BigQueryPlatformRef,
    Field(discriminator="kind"),
]


class SqlType(BaseModel):
    """A SQL column type with native and canonical representations."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw: Annotated[str, Field(min_length=1)]
    canonical: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _accept_string_shorthand(cls, value: Any) -> Any:
        """Turn a bare type string into `{raw, canonical=None}`.

        Args:
            value: The raw input — either a string shorthand or a mapping.

        Returns:
            A mapping suitable for field validation; the input unchanged if not a string.
        """
        if isinstance(value, str):
            return {"raw": value}
        return value

    @classmethod
    def parse(cls, raw: str, dialect: "SQLDialect") -> "SqlType":
        """Build a `SqlType`, canonicalising `raw` in `dialect`.

        Args:
            raw: The native SQL type string.
            dialect: The SQLGlot dialect to parse `raw` in.

        Returns:
            A `SqlType` whose `canonical` is the dialect-neutral SQLGlot rendering, or
            `None` if `raw` is not parseable in `dialect`.
        """
        try:
            canonical = exp.DataType.build(raw, dialect=dialect).sql()
        except SqlglotError:
            return cls(raw=raw, canonical=None)
        return cls(raw=raw, canonical=canonical)

    def __eq__(self, other: object) -> bool:
        """Compare on `canonical` when both sides have one, else on `raw`.

        Returns:
            `True` if the types are equal under that rule, `NotImplemented` if `other`
            is not a `SqlType`.
        """
        if not isinstance(other, SqlType):
            return NotImplemented
        if self.canonical is not None and other.canonical is not None:
            return self.canonical == other.canonical
        return self.raw == other.raw

    def __hash__(self) -> int:
        """Hash on `canonical` when present, else `raw` — consistent with `__eq__`.

        Returns:
            The hash of the comparison key.
        """
        return hash(self.canonical if self.canonical is not None else self.raw)


class Column(BaseModel):
    """A result-set column: name, SQL type, and tri-state nullability."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    type: SqlType
    nullable: bool | None = None


def _reject_duplicate_column_names(names: list[str]) -> None:
    """Raise if any column name repeats.

    Args:
        names: The column names to check, in order.

    Raises:
        ValueError: If a column name appears more than once.
    """
    duplicates = [name for name, count in Counter(names).items() if count > 1]
    if duplicates:
        listed = ", ".join(repr(name) for name in duplicates)
        msg = f"duplicate column name(s): {listed}"
        raise ValueError(msg)


class TypedSchema(RootModel[list[Column]]):
    """An ordered sequence of typed result-set columns with unique names.

    Wraps a `list[Column]` and serialises as a plain JSON array. Engines that report no
    result-column types produce an `UntypedSchema` instead.
    """

    root: list[Column]

    @model_validator(mode="after")
    def _reject_duplicate_names(self) -> "TypedSchema":
        """Reject a repeated column name.

        Returns:
            The validated schema.
        """
        _reject_duplicate_column_names([c.name for c in self.root])
        return self

    def __iter__(self) -> Iterator[Column]:  # ty: ignore[invalid-method-override]
        """Return an iterator over the columns in order."""
        return iter(self.root)

    def __len__(self) -> int:
        """Return the number of columns."""
        return len(self.root)

    def __getitem__(self, index: int) -> Column:
        """Return the column at `index` (positional)."""
        return self.root[index]

    @property
    def names(self) -> list[str]:
        """The column names in order."""
        return [c.name for c in self.root]

    @property
    def types(self) -> list["SqlType"]:
        """The column types in order, positionally aligned with `names`."""
        return [c.type for c in self.root]

    @property
    def types_by_name(self) -> dict[str, "SqlType"]:
        """The column types keyed by name."""
        return {c.name: c.type for c in self.root}


class UntypedSchema(RootModel[list[str]]):
    """An ordered sequence of unique result-column names with no type information.

    Produced by engines whose driver reports no result-column types (e.g. SQLite). Carries
    only names; type comparison against an `UntypedSchema` is inconclusive rather than refuting.
    """

    root: list[str]

    @model_validator(mode="after")
    def _reject_duplicate_names(self) -> "UntypedSchema":
        """Reject a repeated column name.

        Returns:
            The validated schema.
        """
        _reject_duplicate_column_names(self.root)
        return self

    @property
    def names(self) -> list[str]:
        """The column names in order."""
        return list(self.root)


Schema = TypedSchema | UntypedSchema


class UntypedResultSet(BaseModel):
    """Expected outcome as concrete rows without column types: value comparison only."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["untyped_result_set"] = "untyped_result_set"
    rows: list[dict[str, Any]]


class TypedResultSet(BaseModel):
    """Expected outcome as concrete rows plus a schema: value and type comparison."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    kind: Literal["typed_result_set"] = "typed_result_set"
    rows: list[dict[str, Any]]
    schema_: TypedSchema = Field(alias="schema")


class GoldQuery(BaseModel):
    """Expected outcome as a gold/reference query whose executed result IS the expected answer.

    The expected answer is whatever `sql` returns when executed, not literal rows authored
    up front.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["gold_query"] = "gold_query"
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
    expected_type: SqlType


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


def _infer_result_set_kind(value: Any) -> Any:
    """Tag an untagged result-set dict by schema-presence so `kind` may be omitted.

    A `rows` dict with no `kind` becomes `typed_result_set` when it carries a `schema`
    (or `schema_`) key, else `untyped_result_set`. Non-dict, already-tagged, and
    non-result-set inputs pass through unchanged.

    Args:
        value: The raw input to the `Expected` union.

    Returns:
        The input, with `kind` injected when it was an untagged result-set dict.
    """
    if isinstance(value, dict) and "kind" not in value and "rows" in value:
        typed = "schema" in value or "schema_" in value
        return {**value, "kind": "typed_result_set" if typed else "untyped_result_set"}
    return value


_TaggedExpected = Annotated[
    UntypedResultSet | TypedResultSet | GoldQuery | ExpectationSuite,
    Field(discriminator="kind"),
]

Expected = Annotated[_TaggedExpected, BeforeValidator(_infer_result_set_kind)]


class ComparisonConfig(BaseModel):
    """Rules for deciding whether two result sets are equivalent.

    A non-empty `match_key` selects the keyed `FULL OUTER JOIN` comparison: rows are
    aligned on the key columns and compared per remaining column, enabling
    `null_equality="distinct"`, an exact `abs(actual - expected) <= float_tolerance`
    band, and per-column mismatch counts. An empty `match_key` uses the keyless bag
    (multiset) comparison.
    """

    model_config = ConfigDict(extra="forbid")

    column_order: Literal["ignore", "strict"] = "ignore"
    null_equality: Literal["equal", "distinct"] = "equal"
    float_tolerance: Annotated[float, Field(ge=0.0)] = 1e-9
    match_key: list[str] = Field(default_factory=list)


class CostBudget(BaseModel):
    """Per-eval-case ceiling on platform resource consumption."""

    model_config = ConfigDict(extra="forbid")

    max_seconds: Annotated[float, Field(gt=0)] | None = None


class EvalCase(BaseModel):
    """One evaluation case: an input with an expected outcome and a platform to run against."""

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    input: Annotated[str, Field(min_length=1)]
    expected: Expected
    platform: PlatformRef
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)
    cost_budget: CostBudget | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _canonicalize_expected_schema(self) -> "EvalCase":
        """Canonicalise a `TypedResultSet` schema's types in the case's dialect.

        The schema is authored with native type strings whose dialect is the platform's.
        Re-build each `Column.type` via `SqlType.parse` so `canonical` is populated and
        comparison is dialect-free downstream.

        Returns:
            The validated `EvalCase`.
        """
        if isinstance(self.expected, TypedResultSet):
            dialect = self.platform.dialect or self.platform.kind
            self.expected.schema_ = TypedSchema(
                root=[
                    Column(name=c.name, type=SqlType.parse(c.type.raw, dialect), nullable=c.nullable)
                    for c in self.expected.schema_
                ]
            )
        return self

    @model_validator(mode="after")
    def _canonicalize_expectation_types(self) -> "EvalCase":
        """Canonicalise each `ColumnTypeExpectation`'s type in the case's dialect.

        The type is authored as a native string whose dialect is the platform's. Re-build
        `expected_type` via `SqlType.parse` so `canonical` is populated at the ingestion
        boundary and type comparison is dialect-free downstream.

        Returns:
            The validated `EvalCase`.
        """
        if isinstance(self.expected, ExpectationSuite):
            dialect = self.platform.dialect or self.platform.kind
            for expectation in self.expected.expectations:
                if isinstance(expectation, ColumnTypeExpectation):
                    expectation.expected_type = SqlType.parse(expectation.expected_type.raw, dialect)
        return self


class Error(BaseModel):
    """Base type for structured errors."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    message: Annotated[str, Field(min_length=1)]
    cause: Exception | None = Field(default=None, exclude=True, repr=False)


ProviderErrorKind = Literal[
    "timeout",
    "rate_limit",
    "auth",
    "context_window_exceeded",
    "bad_request",
    "api_connection",
    "api_error",
]

SolverErrorKind = ProviderErrorKind | Literal["empty_response", "invalid_structured_output"]


class SolverError(Error):
    """A typed, expected failure from a Solver call."""

    kind: SolverErrorKind
    provider: str | None = None


LlmErrorKind = ProviderErrorKind | Literal["malformed_output"]


class LlmError(Error):
    """A typed, expected failure from an `Llm.complete` call."""

    kind: LlmErrorKind
    provider: str | None = None


class _SolverOutputBase(BaseModel):
    """Usage data shared by solver outcomes."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    latency_seconds: Annotated[float, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SolverSuccess(_SolverOutputBase):
    """SQL produced successfully by a solver."""

    status: Literal["success"] = "success"
    output: Annotated[Sql, Field(min_length=1)]


class SolverFailure(_SolverOutputBase):
    """A solver failure."""

    status: Literal["failure"] = "failure"
    error: SolverError


SolverOutput: TypeAlias = Annotated[SolverSuccess | SolverFailure, Field(discriminator="status")]


ExecutionErrorKind = Literal[
    "query_failed",
    "budget_exceeded",
    "duplicate_columns",
    "type_probe_failed",
    "platform_unavailable",
]


class ExecutionError(Error):
    """A typed failure from running SQL against a platform.

    `condition` carries the driver's error class/code string (e.g. Spark's
    `TABLE_OR_VIEW_NOT_FOUND`) when it exposes one, since not every engine reports SQLSTATE.
    """

    kind: ExecutionErrorKind
    sqlstate: str | None = None
    condition: str | None = None
    params: dict[str, str] | None = None


class NormalizationError(Error):
    """A typed failure from parsing or normalizing SQL for comparison."""

    kind: Literal["parse_failed", "not_single_statement", "normalize_failed", "non_deterministic"]


class _ExecutionResultBase(BaseModel):
    """Measurements shared by successful and failed SQL executions."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    latency_seconds: Annotated[float, Field(ge=0)]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionSuccess(_ExecutionResultBase):
    """A successful SQL execution."""

    status: Literal["success"] = "success"
    rows: list[dict[str, Any]]
    schema_: Schema | None = Field(default=None, alias="schema")


class ExecutionFailure(_ExecutionResultBase):
    """A failed SQL execution."""

    status: Literal["failure"] = "failure"
    error: ExecutionError


ExecutionResult: TypeAlias = Annotated[ExecutionSuccess | ExecutionFailure, Field(discriminator="status")]


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
    sample_missing_rows: list[dict[str, Any]] = Field(default_factory=list)
    sample_extra_rows: list[dict[str, Any]] = Field(default_factory=list)
    missing_columns: list[str] = Field(default_factory=list)
    unexpected_columns: list[str] = Field(default_factory=list)
    type_mismatches: list[TypeMismatch] = Field(default_factory=list)
    column_mismatches: list[ColumnMismatch] = Field(default_factory=list)
    column_order_mismatch: bool = False


class ExpectationOutcome(BaseModel):
    """The result of checking one `Expectation` against an executed result.

    `expected`/`actual` carry the compared scalars (a row count, a column's type
    `raw`); `count` carries the number of offending elements (NULL or duplicate
    values); `sample_rows` carries a bounded sample of the offending rows (empty
    unless the expectation failed and the kind produces one); `detail` is the
    human-readable failure message, `None` when the expectation holds. Which fields
    are populated depends on the expectation kind.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Annotated[str, Field(min_length=1)]
    passed: bool
    column: str | None = None
    expected: str | None = None
    actual: str | None = None
    count: Annotated[int, Field(ge=0)] | None = None
    sample_rows: list[dict[str, Any]] = Field(default_factory=list)
    detail: Annotated[str, Field(min_length=1)] | None = None


class ScoreResult(BaseModel):
    """The outcome of running a Scorer against an EvalCase: a verdict plus diagnostics.

    `score` and `basis` must be absent when `verdict` is `"inconclusive"` — an undecided result
    carries no evidence.
    """

    model_config = ConfigDict(extra="forbid")

    scorer: Annotated[str, Field(min_length=1)]
    verdict: Verdict
    score: Score | None = None
    basis: Basis | None = None
    diff: ResultSetDiff | None = None
    outcomes: list[ExpectationOutcome] = Field(default_factory=list)
    explanation: Annotated[str, Field(min_length=1)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether the verdict is `"pass"`."""
        return self.verdict == "pass"

    @model_validator(mode="after")
    def _inconclusive_carries_no_evidence(self) -> "ScoreResult":
        """Reject a graded `score` or a `basis` on an inconclusive verdict.

        Returns:
            The validated `ScoreResult`.

        Raises:
            ValueError: If `verdict` is `"inconclusive"` and `score` or `basis` is set.
        """
        if self.verdict == "inconclusive" and (self.score is not None or self.basis is not None):
            msg = "an inconclusive ScoreResult cannot carry a graded score or a basis"
            raise ValueError(msg)
        return self


Equivalence = Literal["equivalent", "unknown"]

EquivalenceMethod = Literal["ast"]


class SemanticVerdict(BaseModel):
    """One equivalence check's judgment on whether two queries are equivalent.

    A verdict never carries a diff; a refutation surfaces as a result-set `ScoreResult.diff`.
    """

    model_config = ConfigDict(extra="forbid")

    method: EquivalenceMethod
    equivalence: Equivalence
    detail: Annotated[str, Field(min_length=1)] | None = None
