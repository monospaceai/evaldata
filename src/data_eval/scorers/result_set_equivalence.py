"""`ResultSetEquivalence`: result-set scorer that diffs in-warehouse.

A non-empty `ComparisonConfig.match_key` selects the keyed `FULL OUTER JOIN` path (rows
aligned on the key, compared per column); otherwise the keyless `EXCEPT ALL` bag path runs.
"""

from typing import Any

from data_eval.equivalence import ColumnReconciliation, build_result_set_diff, reconcile_columns
from data_eval.scorers import sql
from data_eval.scorers.context import ScoreContext
from data_eval.scorers.query import QueryRunner
from data_eval.types import (
    ColumnMismatch,
    ComparisonConfig,
    EvalCase,
    ExecutionResult,
    ExpectedResultSet,
    Schema,
    ScoreResult,
    SolverOutput,
    TypeMismatch,
)

SCORER_NAME = "result_set_equivalence"


class ResultSetEquivalence:
    """Scores a case by diffing its executed result set against its `ExpectedResultSet` in SQL."""

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Compare `result` against `case.expected`; pass iff the engine finds them equivalent.

        Column reconciliation and type comparison run in Python; row equivalence is pushed
        into the platform, with authored expected rows materialised as typed literals so the
        engine defines equality. With an empty `match_key`, two `EXCEPT ALL` diffs compute the
        bag difference (and `null_equality="distinct"` is rejected). With a non-empty
        `match_key`, a `FULL OUTER JOIN` aligns rows on the key and compares per column —
        supporting `null_equality="distinct"`, an exact tolerance band, and per-column
        mismatch counts. Only mismatch counts and bounded samples are read back.

        Args:
            case: The eval case, carrying the expected result set, comparison config, and platform.
            output: The solver output (part of the `Scorer` protocol; unused here).
            result: The executed result to compare against the expectation.
            context: The score context, carrying the budget-aware `QueryRunner`.

        Returns:
            A `ScoreResult` that passes when the result set matches the expectation. A failed
            model query, a failed derived query, a non-unique or absent `match_key`, or
            (keyless) `null_equality="distinct"` each yield a failing result with an explanation.

        Raises:
            TypeError: If `case.expected` is not an `ExpectedResultSet`.
        """
        expected = case.expected
        if not isinstance(expected, ExpectedResultSet):
            msg = f"ResultSetEquivalence requires an ExpectedResultSet; got {type(expected).__name__}"
            raise TypeError(msg)

        if result.error is not None:
            return ScoreResult(
                scorer=SCORER_NAME,
                passed=False,
                explanation=f"query execution failed: {result.error}",
            )

        config = case.comparison
        actual_names = _column_names(result.schema_, result.rows)
        expected_names = _column_names(expected.schema_, expected.rows)
        columns = reconcile_columns(actual_names, expected_names, config.column_order)
        type_mismatches = _type_mismatches(result.schema_, expected.schema_, columns.in_both)

        if config.match_key:
            return _keyed_score(expected, result, columns, type_mismatches, config, context.queries)

        if config.null_equality == "distinct":
            return ScoreResult(
                scorer=SCORER_NAME,
                passed=False,
                explanation="null_equality='distinct' requires a match_key (the keyless EXCEPT ALL path treats NULLs as equal)",
            )

        diff_or_error = _diff_rows(expected, columns.in_both, config.float_tolerance, context.queries)
        if isinstance(diff_or_error, str):
            return ScoreResult(scorer=SCORER_NAME, passed=False, explanation=f"query execution failed: {diff_or_error}")
        missing_count, extra_count, sample_missing, sample_extra = diff_or_error

        diff = build_result_set_diff(
            expected_row_count=len(expected.rows),
            actual_row_count=len(result.rows),
            missing_row_count=missing_count,
            extra_row_count=extra_count,
            sample_missing_rows=sample_missing,
            sample_extra_rows=sample_extra,
            columns=columns,
            type_mismatches=type_mismatches,
            column_mismatches=[],
        )
        return ScoreResult(scorer=SCORER_NAME, passed=diff is None, diff=diff)


def _failure(explanation: str) -> ScoreResult:
    """Build a failing `ScoreResult` carrying `explanation` (errors-as-values).

    Args:
        explanation: The human-readable reason the comparison could not pass.

    Returns:
        A failing `ScoreResult` with no diff.
    """
    return ScoreResult(scorer=SCORER_NAME, passed=False, explanation=explanation)


def _keyed_score(
    expected: ExpectedResultSet,
    result: ExecutionResult,
    columns: ColumnReconciliation,
    type_mismatches: list[TypeMismatch],
    config: ComparisonConfig,
    queries: QueryRunner,
) -> ScoreResult:
    """Score the keyed `FULL OUTER JOIN` path: align on `match_key`, compare per column.

    The match key must name shared columns and be unique on both sides; key-only rows become
    missing/extra and key-matched rows that differ on a column populate `column_mismatches`.

    Args:
        expected: The expected result set (rows + optional schema).
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
    numeric = _numeric_columns(expected.schema_, value_columns, dialect)
    expected_rel = sql.expected_relation(expected.rows, expected.schema_, in_both, dialect, None)
    actual_rel = sql.aligned_actual(queries.model_sql, in_both, numeric, dialect, None)

    for relation, side in ((expected_rel, "expected"), (actual_rel, "actual")):
        dupes = queries.scalar(sql.keyed_dupes_count(relation, config.match_key, dialect))
        if dupes.error is not None:
            return _failure(f"query execution failed: {dupes.error}")
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
    if stats.error is not None:
        return _failure(f"query execution failed: {stats.error}")
    counts = list(stats.rows[0].values())
    missing_count = int(counts[0] or 0)
    extra_count = int(counts[1] or 0)
    column_mismatches = [
        ColumnMismatch(column=col, unexpected_count=int(count or 0))
        for col, count in zip(value_columns, counts[2:], strict=True)
        if int(count or 0) > 0
    ]

    samples = _keyed_samples(expected_rel, actual_rel, config.match_key, in_both, missing_count, extra_count, queries)
    if isinstance(samples, str):
        return _failure(f"query execution failed: {samples}")
    sample_missing, sample_extra = samples

    diff = build_result_set_diff(
        expected_row_count=len(expected.rows),
        actual_row_count=len(result.rows),
        missing_row_count=missing_count,
        extra_row_count=extra_count,
        sample_missing_rows=sample_missing,
        sample_extra_rows=sample_extra,
        columns=columns,
        type_mismatches=type_mismatches,
        column_mismatches=column_mismatches,
    )
    return ScoreResult(scorer=SCORER_NAME, passed=diff is None, diff=diff)


_Samples = tuple[list[dict[str, Any]], list[dict[str, Any]]]


def _keyed_samples(
    expected_rel: Any,
    actual_rel: Any,
    match_key: list[str],
    in_both: list[str],
    missing_count: int,
    extra_count: int,
    queries: QueryRunner,
) -> _Samples | str:
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
        `(sample_missing, sample_extra)` on success, or an error message string when a
        derived sample query fails.
    """
    sample_missing: list[dict[str, Any]] = []
    if missing_count:
        run = queries.run(sql.keyed_sample(expected_rel, actual_rel, match_key, in_both, "missing", queries.dialect))
        if run.error is not None:
            return run.error
        sample_missing = run.rows
    sample_extra: list[dict[str, Any]] = []
    if extra_count:
        run = queries.run(sql.keyed_sample(expected_rel, actual_rel, match_key, in_both, "extra", queries.dialect))
        if run.error is not None:
            return run.error
        sample_extra = run.rows
    return (sample_missing, sample_extra)


_RowDiff = tuple[int, int, list[dict[str, Any]], list[dict[str, Any]]]


def _diff_rows(
    expected: ExpectedResultSet,
    in_both: list[str],
    float_tolerance: float,
    queries: QueryRunner,
) -> _RowDiff | str:
    """Compute the bag diff over `in_both` via two `EXCEPT ALL` runs, or return an error string.

    Args:
        expected: The expected result set (rows + optional schema).
        in_both: The shared columns to diff on, in expected order.
        float_tolerance: The absolute tolerance; `> 0` rounds numeric columns before diffing.
        queries: The budget-aware runner used to execute the derived diff queries.

    Returns:
        `(missing_count, extra_count, sample_missing, sample_extra)` on success, where
        `missing` are expected rows absent from actual and `extra` are actual rows absent
        from expected; or an error message string when a derived query fails. With no shared
        columns the diff is empty `(0, 0, [], [])` and no query runs.
    """
    if not in_both:
        return (0, 0, [], [])

    round_scale = sql._round_scale(float_tolerance) if float_tolerance > 0 else None
    numeric = _numeric_columns(expected.schema_, in_both, queries.dialect)
    expected_rel = sql.expected_relation(expected.rows, expected.schema_, in_both, queries.dialect, round_scale)
    actual_rel = sql.aligned_actual(queries.model_sql, in_both, numeric, queries.dialect, round_scale)

    missing = queries.scalar(sql.except_all_count(expected_rel, actual_rel, queries.dialect))
    if missing.error is not None:
        return missing.error
    extra = queries.scalar(sql.except_all_count(actual_rel, expected_rel, queries.dialect))
    if extra.error is not None:
        return extra.error

    missing_count = int(missing.value or 0)
    extra_count = int(extra.value or 0)
    sample_missing: list[dict[str, Any]] = []
    if missing_count:
        run = queries.run(sql.except_all_sample(expected_rel, actual_rel, queries.dialect))
        if run.error is not None:
            return run.error
        sample_missing = run.rows
    sample_extra: list[dict[str, Any]] = []
    if extra_count:
        run = queries.run(sql.except_all_sample(actual_rel, expected_rel, queries.dialect))
        if run.error is not None:
            return run.error
        sample_extra = run.rows
    return (missing_count, extra_count, sample_missing, sample_extra)


def _column_names(schema: Schema | None, rows: list[dict[str, Any]]) -> list[str]:
    """Resolve column names from a schema if present, else the first row's keys.

    Args:
        schema: The result/expected schema, or `None`.
        rows: The rows, used as a fallback for names.

    Returns:
        The column names in order, or `[]` when neither a schema nor any rows are present.
    """
    if schema is not None:
        return schema.names
    if rows:
        return list(rows[0].keys())
    return []


def _type_mismatches(actual: Schema | None, expected: Schema | None, in_both: list[str]) -> list[TypeMismatch]:
    """Compare shared-column types when both schemas are present.

    Args:
        actual: The actual schema, or `None`.
        expected: The expected schema, or `None`.
        in_both: The shared columns to compare, in expected order.

    Returns:
        A `TypeMismatch` per shared column whose actual type differs from the expected type;
        empty when either schema is absent.
    """
    if actual is None or expected is None:
        return []
    actual_types = dict(zip(actual.names, actual.types, strict=True))
    expected_types = dict(zip(expected.names, expected.types, strict=True))
    return [
        TypeMismatch(column=col, expected=expected_types[col].raw, actual=actual_types[col].raw)
        for col in in_both
        if actual_types[col] != expected_types[col]
    ]


def _numeric_columns(schema: Schema | None, in_both: list[str], dialect: sql.Dialect) -> set[str]:
    """Resolve which shared columns are numeric, from the expected schema's types.

    Args:
        schema: The expected schema, or `None` (then no column is treated as numeric).
        in_both: The shared columns to classify.
        dialect: The dialect to parse the column types in.

    Returns:
        The subset of `in_both` whose expected type is numeric.
    """
    if schema is None:
        return set()
    types = dict(zip(schema.names, schema.types, strict=True))
    return {col for col in in_both if col in types and sql._is_numeric_type(types[col].raw, dialect)}
