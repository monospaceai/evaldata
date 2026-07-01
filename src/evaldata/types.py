"""Core public Pydantic types for evaldata."""

from collections import Counter
from collections.abc import Iterator
from typing import Annotated, Any, Literal, NewType

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, RootModel, model_validator
from sqlglot import exp
from sqlglot.errors import SqlglotError

# A SQL string designated as a solver's runnable artifact. Static-only: at runtime it is a
# plain `str`, so it flows transparently into anything that executes SQL; producing one
# requires an explicit `Sql(...)`, which is where the "this string is SQL" claim is made.
Sql = NewType("Sql", str)

# The supported set: dispatch over this Literal is exhaustively type-checked (match/assert_never).
PlatformKind = Literal["duckdb", "postgres", "databricks", "sqlite"]

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

# A scorer-level test outcome: a confirmed pass, a refuted fail, or `"inconclusive"` when the
# test could neither confirm nor refute.
Verdict = Literal["pass", "fail", "inconclusive"]

# Evidence strength of a verdict: sound on all data / observed on the data run / probabilistic judgment.
Basis = Literal["proven", "observed", "judged"]

# A normalized graded magnitude in `[0.0, 1.0]`, higher-is-better.
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class PlatformRef(BaseModel):
    """Serializable reference to a configured data platform connection."""

    model_config = ConfigDict(extra="forbid")

    name: Annotated[str, Field(min_length=1)]
    kind: PlatformKind
    dialect: SQLDialect | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class SqlType(BaseModel):
    """A SQL column type, canonicalised through SQLGlot for safe equality.

    `raw` is the native string the platform or author produced (truthful, for diffs).
    `canonical` is the dialect-neutral SQLGlot rendering used for comparison, or `None`
    when SQLGlot cannot parse the type. Equality compares `canonical` when both sides
    have one, else falls back to `raw`.

    Accepts a plain string as authoring shorthand: `"BIGINT"` becomes a `SqlType` with
    `raw="BIGINT"` and `canonical=None`. Canonicalisation happens eagerly at the
    ingestion boundary (adapters and the `EvalCase` validator), never at compare time.
    """

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


class TypedSchema(RootModel[list[Column]]):
    """An ordered, duplicate-faithful sequence of typed result-set columns.

    Wraps a `list[Column]` and serialises as a plain JSON array. Name lookup is not
    offered: result sets may repeat column names, so the convenience accessors are
    positional and duplicate-safe. Engines that report no result-column types produce an
    `UntypedSchema` instead.
    """

    root: list[Column]

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
        """The column names in order (duplicate-faithful)."""
        return [c.name for c in self.root]

    @property
    def types(self) -> list["SqlType"]:
        """The column types in order, positionally aligned with `names`."""
        return [c.type for c in self.root]


class UntypedSchema(RootModel[list[str]]):
    """An ordered sequence of result-column names with no type information.

    Produced by engines whose driver reports no result-column types (e.g. SQLite). Carries
    only names; type comparison against an `UntypedSchema` abstains rather than refuting.
    """

    root: list[str]

    @property
    def names(self) -> list[str]:
        """The column names in order (duplicate-faithful)."""
        return list(self.root)


# A result set's schema: typed (every column has a `SqlType`) or untyped (names only).
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
    (`EXCEPT ALL`) comparison.
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
    """One AI-evaluation case: an input with an expected outcome and a platform to run against."""

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
        comparison is dialect-free downstream. Duplicate column names are rejected: rows
        are matched by name during comparison, so a repeated name is unresolvable.

        Returns:
            The validated `EvalCase`.

        Raises:
            ValueError: If the expected schema declares a column name more than once.
        """
        if isinstance(self.expected, TypedResultSet):
            names = self.expected.schema_.names
            duplicates = [name for name, count in Counter(names).items() if count > 1]
            if duplicates:
                listed = ", ".join(repr(name) for name in duplicates)
                msg = f"expected schema has duplicate column name(s): {listed}"
                raise ValueError(msg)
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
    """Base for the typed error types.

    Holds the fields every typed error shares. Subclasses add a `kind` discriminator and any
    domain-specific structured fields (an `ExecutionError`'s `sqlstate`, a `SolverError`'s
    `provider`). `cause` keeps the original exception — and its traceback — for in-process
    debugging and logging; it is excluded from serialization, so reports carry only the
    structured surface.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    message: Annotated[str, Field(min_length=1)]
    cause: Exception | None = Field(default=None, exclude=True, repr=False)


# The provider-call failures shared by every LLM-backed call: a 1:1 reflection of the
# litellm exception hierarchy the LLM seam translates.
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


class SolverOutput(BaseModel):
    """A Solver's output: either a successful `output` artifact or an `error`.

    Exactly one of `output`/`error` is set. For SQL solvers, `output` is the SQL to run.
    """

    model_config = ConfigDict(extra="forbid")

    output: Annotated[Sql, Field(min_length=1)] | None = None
    error: SolverError | None = None
    prompt_tokens: Annotated[int, Field(ge=0)] | None = None
    completion_tokens: Annotated[int, Field(ge=0)] | None = None
    latency_seconds: Annotated[float, Field(ge=0)] | None = None
    cost_usd: Annotated[float, Field(ge=0)] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _exactly_one_of_output_or_error(self) -> "SolverOutput":
        """Enforce that exactly one of `output`/`error` is set.

        Returns:
            The validated `SolverOutput`.

        Raises:
            ValueError: If Pydantic validation fails.
        """
        if (self.output is None) == (self.error is None):
            msg = "SolverOutput requires exactly one of 'output' or 'error' to be set"
            raise ValueError(msg)
        return self


ExecutionErrorKind = Literal["query_failed", "budget_exceeded", "duplicate_columns", "type_probe_failed"]


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


class ExecutionResult(BaseModel):
    """The result of running SQL against a platform: returned rows plus execution measurements."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, serialize_by_alias=True)

    rows: list[dict[str, Any]]
    schema_: Schema | None = Field(default=None, alias="schema")
    latency_seconds: Annotated[float, Field(ge=0)]
    error: ExecutionError | None = None
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
    # Bounded samples of the differing rows, capped by the engine so large mismatches
    # stay readable. `*_row_count` give the full magnitude; these give concrete examples.
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


# A semantic check either confirms equivalence or cannot decide; it never refutes, so "unknown"
# means "could not confirm".
Equivalence = Literal["equivalent", "unknown"]

# The kinds of equivalence-deciding technique.
EquivalenceMethod = Literal["ast"]


class SemanticVerdict(BaseModel):
    """One equivalence check's judgment on whether two queries are equivalent.

    A verdict never carries a diff; a refutation surfaces as a result-set `ScoreResult.diff`.
    """

    model_config = ConfigDict(extra="forbid")

    method: EquivalenceMethod
    equivalence: Equivalence
    detail: Annotated[str, Field(min_length=1)] | None = None
