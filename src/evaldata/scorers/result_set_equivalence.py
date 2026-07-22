"""`ResultSetEquivalence`: result-set scorer that diffs in-warehouse.

A non-empty `ComparisonConfig.match_key` selects the keyed `FULL OUTER JOIN` path (rows
aligned on the key, compared per column); otherwise the keyless bag-difference path runs.
The expected side is either authored rows materialised as typed literals, or a `GoldQuery`
embedded as a subquery so the reference query never egresses its rows into Python.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sqlglot import exp

from evaldata.equivalence import ColumnReconciliation, build_result_set_diff, reconcile_columns
from evaldata.scorers import sql
from evaldata.scorers.base import misconfigured
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.query import QueryRunner, ScalarFailure
from evaldata.types import (
    ColumnMismatch,
    ComparisonConfig,
    EvalCase,
    ExecutionError,
    ExecutionFailure,
    ExecutionResult,
    ExecutionSuccess,
    GoldQuery,
    Schema,
    ScoreResult,
    SolverSuccess,
    TypedResultSet,
    TypedSchema,
    TypeMismatch,
    UntypedResultSet,
)

SCORER_NAME = "result_set_equivalence"

# A builder for the expected relation over the shared columns, given the columns (in expected
# order), the numeric subset, and a `ROUND` scale (`None` for no rounding). Authored rows close
# over their literals; a gold query closes over its subquery SQL.
_ExpectedRelation = Callable[[list[str], set[str], int | None], exp.Query]


@dataclass(frozen=True)
class _ExpectedSource:
    """The expected side of the comparison, resolved from authored rows or a gold query.

    Attributes:
        schema_: The expected schema; untyped or `None` means no types are available to compare.
        relation: Builds the expected relation over the shared columns.
    """

    schema_: Schema | None
    names: list[str]
    row_count: int
    relation: _ExpectedRelation


class ResultSetEquivalence:
    """Scores a case by diffing its executed result set against its expected result set in SQL."""

    def score(
        self, case: EvalCase, output: SolverSuccess, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Compare `result` against `case.expected`; pass iff the engine finds them equivalent.

        Column reconciliation and type comparison run in Python; row equivalence is pushed
        into the platform. For authored rows the expected side is materialised as typed
        literals; for a `GoldQuery` it is the reference query embedded as a subquery, whose
        schema is discovered with a zero-row execution and whose rows never reach Python.
        With an empty `match_key`, two bag-difference queries compute missing and extra rows (and
        `null_equality="distinct"` is rejected). With a non-empty `match_key`, a
        `FULL OUTER JOIN` aligns rows on the key and compares per column — supporting
        `null_equality="distinct"`, an exact tolerance band, and per-column mismatch counts.
        Only mismatch counts and bounded samples are read back.

        Args:
            case: The eval case, carrying the expected result set, comparison config, and platform.
            output: The solver output (part of the `Scorer` protocol; unused here).
            result: The executed result to compare against the expectation.
            context: The score context, carrying the budget-aware `QueryRunner`.

        Returns:
            A `ScoreResult` that passes when the result set matches the expectation. A failed
            model query, a failed gold query, a failed derived query, a non-unique or absent
            `match_key`, or (keyless) `null_equality="distinct"` each yield a failing result
            with an explanation; an `expected` of the wrong kind yields an inconclusive result.
        """
        expected = case.expected
        if not isinstance(expected, UntypedResultSet | TypedResultSet | GoldQuery):
            return misconfigured(SCORER_NAME, expected, "an UntypedResultSet, TypedResultSet, or GoldQuery")

        if isinstance(result, ExecutionFailure):
            return ScoreResult(
                scorer=SCORER_NAME,
                verdict="fail",
                explanation=f"query execution failed: {result.error.message}",
            )

        source = _resolve_expected(expected, context.queries)
        if isinstance(source, ScoreResult):
            return source

        config = case.comparison
        actual_schema = result.schema_
        # Type comparison only applies to typed schemas; an untyped actual (e.g. SQLite) abstains.
        actual_typed = actual_schema if isinstance(actual_schema, TypedSchema) else None
        if isinstance(expected, TypedResultSet) and actual_typed is not None:
            resolved = context.queries.resolved_schema(actual_typed, context.queries.model_sql)
            if isinstance(resolved, ExecutionError):
                return _failure(f"could not resolve column types for type comparison: {resolved.message}")
            actual_typed = resolved
        actual_names = _column_names(actual_schema, result.rows)
        columns = reconcile_columns(actual_names, source.names, config.column_order)
        source_typed = source.schema_ if isinstance(source.schema_, TypedSchema) else None
        type_mismatches = _type_mismatches(actual_typed, source_typed, columns.in_both)

        if config.match_key:
            return _keyed_score(source, result, columns, type_mismatches, config, context.queries)

        if config.null_equality == "distinct":
            return ScoreResult(
                scorer=SCORER_NAME,
                verdict="fail",
                explanation="null_equality='distinct' requires a match_key (the keyless bag path treats NULLs as equal)",
            )

        diff_or_error = _diff_rows(source, columns.in_both, config.float_tolerance, context.queries)
        if isinstance(diff_or_error, ExecutionError):
            return ScoreResult(
                scorer=SCORER_NAME, verdict="fail", explanation=f"query execution failed: {diff_or_error.message}"
            )
        missing_count, extra_count, sample_missing, sample_extra = diff_or_error

        diff = build_result_set_diff(
            expected_row_count=source.row_count,
            actual_row_count=len(result.rows),
            missing_row_count=missing_count,
            extra_row_count=extra_count,
            sample_missing_rows=sample_missing,
            sample_extra_rows=sample_extra,
            columns=columns,
            type_mismatches=type_mismatches,
            column_mismatches=[],
        )
        return ScoreResult(scorer=SCORER_NAME, verdict="pass" if diff is None else "fail", basis="observed", diff=diff)


def _gold_failure(error: ExecutionError) -> ScoreResult:
    """Return a failing `ScoreResult` for a failed gold query, tagged `metadata["gold_query_failed"]=True`."""
    return ScoreResult(
        scorer=SCORER_NAME,
        verdict="fail",
        explanation=f"gold query failed: {error.message}",
        metadata={"gold_query_failed": True},
    )


def _resolve_expected(
    expected: UntypedResultSet | TypedResultSet | GoldQuery, queries: QueryRunner
) -> _ExpectedSource | ScoreResult:
    """Resolve the expected side from authored rows or a gold query.

    Authored rows carry their own schema, names, and count, and materialise as typed literals.
    A gold query discovers its schema with a zero-row execution and its row count with a
    `COUNT(*)`, then embeds itself as a subquery; either gold-attributable query failing yields
    a failing `ScoreResult` rather than raising.

    Args:
        expected: The case's expected result set.
        queries: The budget-aware runner used to discover the gold schema and count.

    Returns:
        An `_ExpectedSource` on success, or a failing `ScoreResult` when a gold query fails.
    """
    if isinstance(expected, GoldQuery):
        return _resolve_gold(expected, queries)

    schema_ = expected.schema_ if isinstance(expected, TypedResultSet) else None
    rows = expected.rows
    names = _column_names(schema_, rows)

    def relation(in_both: list[str], numeric: set[str], round_scale: int | None) -> exp.Query:  # noqa: ARG001
        return sql.expected_relation(rows, schema_, in_both, queries.dialect, round_scale)

    return _ExpectedSource(schema_=schema_, names=names, row_count=len(rows), relation=relation)


def _resolve_gold(expected: GoldQuery, queries: QueryRunner) -> _ExpectedSource | ScoreResult:
    """Discover a gold query's schema and row count, then build its expected source.

    Args:
        expected: The gold query.
        queries: The budget-aware runner used to discover the gold schema and count.

    Returns:
        An `_ExpectedSource` whose relation embeds the gold query as a subquery, or a failing
        `ScoreResult` when the schema-discovery or count query errors.
    """
    gold_sql = sql.Sql(expected.sql)
    probe = queries.run(sql.gold_schema_probe(gold_sql, queries.dialect))
    if isinstance(probe, ExecutionFailure):
        return _gold_failure(probe.error)
    schema_ = probe.schema_
    names = schema_.names if schema_ is not None else []

    count = queries.scalar(sql.row_count(gold_sql, queries.dialect))
    if isinstance(count, ScalarFailure):
        return _gold_failure(count.error)
    row_count = int(count.value or 0)

    def relation(in_both: list[str], numeric: set[str], round_scale: int | None) -> exp.Query:
        return sql.gold_expected(gold_sql, in_both, numeric, queries.dialect, round_scale)

    return _ExpectedSource(schema_=schema_, names=names, row_count=row_count, relation=relation)


def _failure(explanation: str) -> ScoreResult:
    """Return a failing `ScoreResult` carrying `explanation`."""
    return ScoreResult(scorer=SCORER_NAME, verdict="fail", explanation=explanation)


def _keyed_score(
    source: _ExpectedSource,
    result: ExecutionSuccess,
    columns: ColumnReconciliation,
    type_mismatches: list[TypeMismatch],
    config: ComparisonConfig,
    queries: QueryRunner,
) -> ScoreResult:
    """Score the keyed `FULL OUTER JOIN` path: align on `match_key`, compare per column.

    The match key must name shared columns and be unique on both sides; key-only rows become
    missing/extra and key-matched rows that differ on a column populate `column_mismatches`.

    Args:
        source: The resolved expected side (schema, names, count, relation builder).
        result: The executed actual result (for row-count reporting).
        columns: The reconciliation of actual against expected column names.
        type_mismatches: Per-column type differences over the shared columns.
        config: The comparison config, carrying `match_key`, `null_equality`, `float_tolerance`.
        queries: The budget-aware runner used to execute the derived diff queries.

    Returns:
        A passing `ScoreResult` when aligned rows match, else a failing one. An absent or
        non-unique key, or a failed derived query, each yield a failing result.
    """
    in_both = columns.in_both
    shared = set(in_both)
    absent = [key for key in config.match_key if key not in shared]
    if absent:
        listed = ", ".join(repr(key) for key in absent)
        return _failure(f"match_key column(s) not present in both result sets: {listed}")

    dialect = queries.dialect
    value_columns = [col for col in in_both if col not in set(config.match_key)]
    numeric = _numeric_columns(source.schema_, value_columns, dialect)
    expected_rel = source.relation(in_both, numeric, None)
    actual_rel = sql.aligned_actual(queries.model_sql, in_both, numeric, dialect, None)

    for relation, side in ((expected_rel, "expected"), (actual_rel, "actual")):
        dupes = queries.scalar(sql.keyed_dupes_count(relation, config.match_key, dialect))
        if isinstance(dupes, ScalarFailure):
            return _failure(f"query execution failed: {dupes.error.message}")
        if int(dupes.value or 0) > 0:
            return _failure(
                f"match_key is not unique in the {side} result set; omit match_key to compare with bag semantics"
            )

    stats = queries.run(
        sql.keyed_diff_stats(
            expected_rel,
            actual_rel,
            config.match_key,
            value_columns,
            numeric,
            config.null_equality,
            config.float_tolerance,
            in_both,
            dialect,
        )
    )
    if isinstance(stats, ExecutionFailure):
        return _failure(f"query execution failed: {stats.error.message}")
    row = stats.rows[0]
    missing_count = int(row["missing"] or 0)
    extra_count = int(row["extra"] or 0)
    column_mismatches = [
        ColumnMismatch(column=col, unexpected_count=int(row[sql.keyed_mismatch_alias(index)] or 0))
        for index, col in enumerate(value_columns)
        if int(row[sql.keyed_mismatch_alias(index)] or 0) > 0
    ]

    samples = _keyed_samples(expected_rel, actual_rel, config.match_key, in_both, missing_count, extra_count, queries)
    if isinstance(samples, ExecutionError):
        return _failure(f"query execution failed: {samples.message}")
    sample_missing, sample_extra = samples

    diff = build_result_set_diff(
        expected_row_count=source.row_count,
        actual_row_count=len(result.rows),
        missing_row_count=missing_count,
        extra_row_count=extra_count,
        sample_missing_rows=sample_missing,
        sample_extra_rows=sample_extra,
        columns=columns,
        type_mismatches=type_mismatches,
        column_mismatches=column_mismatches,
    )
    return ScoreResult(scorer=SCORER_NAME, verdict="pass" if diff is None else "fail", basis="observed", diff=diff)


_Samples = tuple[list[dict[str, Any]], list[dict[str, Any]]]


def _keyed_samples(
    expected_rel: Any,
    actual_rel: Any,
    match_key: list[str],
    in_both: list[str],
    missing_count: int,
    extra_count: int,
    queries: QueryRunner,
) -> _Samples | ExecutionError:
    """Read bounded samples of the key-only buckets, only for non-empty buckets.

    Args:
        expected_rel: The expected relation over `in_both`.
        actual_rel: The actual relation over `in_both`.
        match_key: The match-key columns aligned on.
        in_both: The shared columns, in expected order.
        missing_count: The count of key-only-in-expected rows (sampled only when positive).
        extra_count: The count of key-only-in-actual rows (sampled only when positive).
        queries: The budget-aware runner used to execute the derived sample queries.

    Returns:
        `(sample_missing, sample_extra)` on success, or an `ExecutionError` when a derived
        sample query fails.
    """
    sample_missing: list[dict[str, Any]] = []
    if missing_count:
        run = queries.run(sql.keyed_sample(expected_rel, actual_rel, match_key, in_both, "missing", queries.dialect))
        if isinstance(run, ExecutionFailure):
            return run.error
        sample_missing = run.rows
    sample_extra: list[dict[str, Any]] = []
    if extra_count:
        run = queries.run(sql.keyed_sample(expected_rel, actual_rel, match_key, in_both, "extra", queries.dialect))
        if isinstance(run, ExecutionFailure):
            return run.error
        sample_extra = run.rows
    return (sample_missing, sample_extra)


_RowDiff = tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]


def _diff_rows(
    source: _ExpectedSource,
    in_both: list[str],
    float_tolerance: float,
    queries: QueryRunner,
) -> _RowDiff | ExecutionError:
    """Compute the bag diff over `in_both` via two portable bag-difference queries.

    Args:
        source: The resolved expected side (schema, relation builder).
        in_both: The shared columns to diff on, in expected order.
        float_tolerance: The absolute tolerance; `> 0` rounds numeric columns before diffing.
        queries: The budget-aware runner used to execute the derived diff queries.

    Returns:
        `(missing_count, extra_count, sample_missing, sample_extra)` on success, where
        `missing` are expected rows absent from actual and `extra` are actual rows absent
        from expected; or an `ExecutionError` when a derived query fails. With no shared
        columns the diff is empty `(0, 0, [], [])` and no query runs.
    """
    if not in_both:
        return (0, 0, [], [])

    round_scale = sql.round_scale(float_tolerance) if float_tolerance > 0 else None
    numeric = _numeric_columns(source.schema_, in_both, queries.dialect)
    expected_rel = source.relation(in_both, numeric, round_scale)
    actual_rel = sql.aligned_actual(queries.model_sql, in_both, numeric, queries.dialect, round_scale)

    missing = queries.scalar(sql.bag_diff_count(expected_rel, actual_rel, in_both, queries.dialect))
    if isinstance(missing, ScalarFailure):
        return missing.error
    extra = queries.scalar(sql.bag_diff_count(actual_rel, expected_rel, in_both, queries.dialect))
    if isinstance(extra, ScalarFailure):
        return extra.error

    missing_count = int(missing.value or 0)
    extra_count = int(extra.value or 0)
    sample_missing: list[dict[str, Any]] = []
    if missing_count:
        run = queries.run(sql.bag_diff_sample(expected_rel, actual_rel, in_both, queries.dialect))
        if isinstance(run, ExecutionFailure):
            return run.error
        sample_missing = run.rows
    sample_extra: list[dict[str, Any]] = []
    if extra_count:
        run = queries.run(sql.bag_diff_sample(actual_rel, expected_rel, in_both, queries.dialect))
        if isinstance(run, ExecutionFailure):
            return run.error
        sample_extra = run.rows
    return (missing_count, extra_count, sample_missing, sample_extra)


def _column_names(schema: Schema | None, rows: list[dict[str, Any]]) -> list[str]:
    """Resolve column names from a schema if present, else the first row's keys.

    Args:
        schema: The result schema (typed or untyped), or `None`.
        rows: The rows, used as a fallback for names.

    Returns:
        The column names in order, or `[]` when neither a schema nor any rows are present.
    """
    if schema is not None:
        return schema.names
    if rows:
        return list(rows[0].keys())
    return []


def _type_mismatches(
    actual: TypedSchema | None, expected: TypedSchema | None, in_both: list[str]
) -> list[TypeMismatch]:
    """Compare shared-column types when both schemas are typed.

    An untyped actual or expected side arrives as `None` — type comparison abstains rather than
    refuting when types are absent (e.g. SQLite reports no result types).

    Args:
        actual: The actual typed schema, or `None`.
        expected: The expected typed schema, or `None`.
        in_both: The shared columns to compare, in expected order.

    Returns:
        A `TypeMismatch` per shared column whose actual type differs from the expected type;
        empty when either schema is absent.
    """
    if actual is None or expected is None:
        return []
    actual_types = actual.types_by_name
    expected_types = expected.types_by_name
    return [
        TypeMismatch(column=col, expected=expected_types[col].raw, actual=actual_types[col].raw)
        for col in in_both
        if actual_types[col] != expected_types[col]
    ]


def _numeric_columns(schema: Schema | None, in_both: list[str], dialect: sql.Dialect) -> set[str]:
    """Resolve which shared columns are numeric, from the expected schema's types.

    Args:
        schema: The expected schema; only a `TypedSchema` carries types — an untyped or absent
            schema treats no column as numeric.
        in_both: The shared columns to classify.
        dialect: The dialect to parse the column types in.

    Returns:
        The subset of `in_both` whose expected type is numeric.
    """
    if not isinstance(schema, TypedSchema):
        return set()
    types = schema.types_by_name
    return {col for col in in_both if col in types and sql.is_numeric_type(types[col].raw, dialect)}
