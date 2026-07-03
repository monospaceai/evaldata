"""`ExecutionAccuracy`: the text-to-SQL execution-accuracy (EX) oracle.

Runs the model's SQL and the gold query against the platform, fetches both result sets into
Python, and compares them. Comparison semantics are configurable: row order (`row_order`),
duplicate-row handling (`multiplicity`), and how columns are aligned (`column_alignment`). The
defaults compare columns positionally with bag semantics, order-sensitive only when the gold
query carries a top-level `ORDER BY`.
"""

from collections import Counter
from typing import Any, Literal

import sqlglot
from sqlglot.errors import SqlglotError

from evaldata.equivalence.rows import compare_rows
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    EvalCase,
    ExecutionError,
    ExecutionResult,
    GoldQuery,
    ResultSetDiff,
    ScoreResult,
    SolverOutput,
    Sql,
)

SCORER_NAME = "execution_accuracy"

# Differing rows to carry as concrete samples in the diff; counts give the full magnitude.
_SAMPLE_CAP = 5

_Row = dict[str, Any]
_Tuple = tuple[Any, ...]


class ExecutionAccuracy:
    """Scores a case by execution accuracy: does the model's SQL return the gold query's rows?"""

    def __init__(
        self,
        *,
        row_order: Literal["when_ordered", "ignore"] = "when_ordered",
        multiplicity: Literal["multiset", "set"] = "multiset",
        column_alignment: Literal["by_position", "by_value"] = "by_position",
    ) -> None:
        """Configure the comparison semantics.

        Args:
            row_order: `"when_ordered"` (default) makes the comparison order-sensitive iff the
                gold query carries a top-level `ORDER BY`; `"ignore"` always compares
                order-insensitively.
            multiplicity: `"multiset"` (default) requires duplicate rows to match by count (bag
                semantics); `"set"` compares distinct rows (set equality).
            column_alignment: `"by_position"` (default) compares columns positionally;
                `"by_value"` ignores column order by searching for a column permutation whose
                values make the result sets match. Use `"by_value"` when the gold and model
                queries label columns differently, so columns can only be aligned by content,
                not by name; it requires both result sets to have the same number of columns.
        """
        self._row_order = row_order
        self._multiplicity = multiplicity
        self._column_alignment = column_alignment

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Compare the model result against the gold query's executed rows.

        Args:
            case: The eval case; its `expected` must be a `GoldQuery`.
            output: The solver output (part of the `Scorer` protocol; unused here).
            result: The executed model result to score.
            context: The score context, carrying the budget-aware `QueryRunner`.

        Returns:
            A passing `ScoreResult` (`basis="observed"`) when the result sets match under the
            configured semantics, else a failing one carrying a `ResultSetDiff`. A failed model
            query, or a failed gold query (`metadata["gold_query_failed"]`), yields a failing
            result with an explanation.

        Raises:
            TypeError: If `case.expected` is not a `GoldQuery`.
        """
        expected = case.expected
        if not isinstance(expected, GoldQuery):
            msg = f"ExecutionAccuracy requires a GoldQuery expected; got {type(expected).__name__}"
            raise TypeError(msg)

        if result.error is not None:
            return _failure(f"query execution failed: {result.error.message}")

        gold = context.queries.run(Sql(expected.sql))
        if gold.error is not None:
            return _gold_failure(gold.error)

        order_sensitive = self._order_sensitive(expected.sql, context.queries.dialect)
        actual_tuples = [tuple(row.values()) for row in result.rows]
        gold_tuples = [tuple(row.values()) for row in gold.rows]
        passed = compare_rows(
            actual_tuples,
            gold_tuples,
            order_sensitive=order_sensitive,
            multiplicity=self._multiplicity,
            column_alignment=self._column_alignment,
        )
        # The diff is positional, so it cannot describe a `by_value` permutation match; attach it
        # only on failure, where it diagnoses the mismatch.
        diff = None if passed else self._diff(result.rows, gold.rows, actual_tuples, gold_tuples)
        return ScoreResult(scorer=SCORER_NAME, verdict="pass" if passed else "fail", basis="observed", diff=diff)

    def _order_sensitive(self, gold_sql: str, dialect: Dialect) -> bool:
        """Whether the comparison should respect row order, per `row_order` and the gold query.

        Args:
            gold_sql: The gold query text.
            dialect: The dialect to parse the gold query in.

        Returns:
            `True` when `row_order="when_ordered"` and the gold query's top-level statement
            carries an `ORDER BY` (a window `OVER (ORDER BY …)` does not count); always `False`
            when `row_order="ignore"` or the query cannot be parsed.
        """
        if self._row_order == "ignore":
            return False
        try:
            parsed = sqlglot.parse_one(gold_sql, dialect=dialect)
        except SqlglotError:
            return False
        return parsed is not None and parsed.args.get("order") is not None

    def _diff(
        self, actual_rows: list[_Row], gold_rows: list[_Row], actual: list[_Tuple], gold: list[_Tuple]
    ) -> ResultSetDiff:
        """Build a diagnostic `ResultSetDiff` from the row difference.

        The diff is positional regardless of `column_alignment`; it is diagnostic only.

        Args:
            actual_rows: The model result rows (for samples, name-keyed).
            gold_rows: The gold query rows (for samples, name-keyed).
            actual: The model result rows as positional tuples.
            gold: The gold query rows as positional tuples.

        Returns:
            A `ResultSetDiff` with expected/actual counts, missing/extra counts, and bounded
            samples of the differing rows. Under `multiplicity="set"` the missing/extra signals
            are computed over distinct rows so they agree with a set verdict. Column-level
            fields are left empty (EX compares values positionally, not by column).
        """
        if self._multiplicity == "set":
            missing = Counter(set(gold) - set(actual))
            extra = Counter(set(actual) - set(gold))
        else:
            missing = Counter(gold) - Counter(actual)
            extra = Counter(actual) - Counter(gold)
        return ResultSetDiff(
            expected_row_count=len(gold),
            actual_row_count=len(actual),
            missing_row_count=sum(missing.values()),
            extra_row_count=sum(extra.values()),
            sample_missing_rows=_samples(gold_rows, gold, missing),
            sample_extra_rows=_samples(actual_rows, actual, extra),
        )


def _samples(rows: list[_Row], tuples: list[_Tuple], wanted: Counter[_Tuple]) -> list[_Row]:
    """Pick up to `_SAMPLE_CAP` of `rows` whose positional tuple is in the `wanted` multiset.

    Args:
        rows: The name-keyed rows, aligned with `tuples`.
        tuples: The positional tuples, aligned with `rows`.
        wanted: The multiset of tuples to sample (a `Counter` of difference rows).

    Returns:
        Up to `_SAMPLE_CAP` rows drawn from `rows`, respecting the multiplicity in `wanted`.
    """
    remaining = Counter(wanted)
    samples: list[_Row] = []
    for row, key in zip(rows, tuples, strict=True):
        if len(samples) >= _SAMPLE_CAP:
            break
        if remaining.get(key, 0) > 0:
            samples.append(row)
            remaining[key] -= 1
    return samples


def _failure(explanation: str) -> ScoreResult:
    """Return a failing `ScoreResult` carrying `explanation`."""
    return ScoreResult(scorer=SCORER_NAME, verdict="fail", explanation=explanation)


def _gold_failure(error: ExecutionError) -> ScoreResult:
    """Return a failing `ScoreResult` for a failed gold query, tagged `metadata["gold_query_failed"]=True`."""
    return ScoreResult(
        scorer=SCORER_NAME,
        verdict="fail",
        explanation=f"gold query failed: {error.message}",
        metadata={"gold_query_failed": True},
    )
