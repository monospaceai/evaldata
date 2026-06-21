"""`SemanticEquivalence`: query-vs-query equivalence via a sequence of pluggable checks.

Each `EquivalenceCheck` returns a three-valued `SemanticVerdict`; the scorer runs the checks
in order and stops at the first decisive verdict, so a cheap structural confirmation can skip
the warehouse entirely. `AstEquivalence` normalizes both queries' syntax trees with SQLGlot
and compares them: it only ever confirms, never refutes; portable and execution-free.
"""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.simplify import simplify

from evaldata.equivalence.semantic import combine
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.result_set_equivalence import ResultSetEquivalence
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    EvalCase,
    ExecutionResult,
    GoldQuery,
    NormalizationError,
    ScoreResult,
    SemanticEquivalenceMethod,
    SemanticVerdict,
    SolverOutput,
    Sql,
)

SCORER_NAME = "semantic_equivalence"


@runtime_checkable
class EquivalenceCheck(Protocol):
    """One way of deciding query-vs-query equivalence; returns a verdict, never raises.

    A check that cannot decide returns an `"unknown"` verdict, and emits only the values it can
    assert with certainty (a structural check confirms or abstains, never refutes).
    """

    name: SemanticEquivalenceMethod

    def judge(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> SemanticVerdict:
        """Judge whether the model's query is equivalent to the case's gold query.

        Args:
            case: The eval case, whose `expected` carries the gold query.
            output: The solver output.
            result: The already-executed model result (reused by execution-based checks).
            context: The score context, carrying the model SQL, dialect, and query runner.

        Returns:
            A `SemanticVerdict` describing the equivalence decision for this check.
        """
        ...


class AstEquivalence:
    """Confirms equivalence when both queries normalize to the same SQLGlot syntax tree.

    Matching normalized trees yield `"equivalent"`; anything else (trees differ, a parse
    failure, or input that is not exactly one statement) yields `"unknown"`, never
    `"not_equivalent"`. The normalization is conservative and schema-free, so some true
    equivalences (e.g. commutative arithmetic over columns) fall through as `"unknown"`.
    """

    name: SemanticEquivalenceMethod = "ast"

    def judge(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> SemanticVerdict:
        """Compare the model and gold queries' normalized syntax trees.

        Args:
            case: The eval case; `expected` must be a `GoldQuery` to compare against.
            output: The solver output (unused).
            result: The executed model result (unused; this check touches no data).
            context: The score context, supplying the model SQL and dialect.

        Returns:
            `"equivalent"` when the normalized trees match, else `"unknown"`.
        """
        gold = case.expected
        if not isinstance(gold, GoldQuery):
            return _unknown("expected is not a gold query")
        dialect = context.queries.dialect
        model_tree = _normalize(context.queries.model_sql, dialect)
        if isinstance(model_tree, NormalizationError):
            return _unknown(f"model query: {model_tree.message}")
        gold_tree = _normalize(Sql(gold.sql), dialect)
        if isinstance(gold_tree, NormalizationError):
            return _unknown(f"gold query: {gold_tree.message}")
        if model_tree == gold_tree:
            return SemanticVerdict(
                method="ast", equivalence="equivalent", detail="queries normalize to the same syntax tree"
            )
        return _unknown("normalized syntax trees differ (inconclusive)")


def _unknown(detail: str) -> SemanticVerdict:
    """Build an `"unknown"` AST verdict carrying `detail`.

    Args:
        detail: The human-readable reason the check could not confirm equivalence.

    Returns:
        A `SemanticVerdict` with `method="ast"` and `equivalence="unknown"`.
    """
    return SemanticVerdict(method="ast", equivalence="unknown", detail=detail)


def _normalize(sql: Sql, dialect: Dialect) -> exp.Expression | NormalizationError:
    """Parse and normalize `sql` into a comparable expression, or return a `NormalizationError`.

    The normalization is conservative, schema-free, and truth-preserving. Returns a
    `NormalizationError` (rather than raising) when `sql` does not parse or is not a single statement.

    Args:
        sql: The query to normalize.
        dialect: The SQLGlot dialect to parse and normalize in (the same for both queries).

    Returns:
        The normalized expression, or a `NormalizationError`.
    """
    try:
        statements = sqlglot.parse(sql, dialect=dialect)
    except SqlglotError as error:
        return NormalizationError(kind="parse_failed", message=f"could not parse ({error})", cause=error)
    parsed = [statement for statement in statements if statement is not None]
    if len(parsed) != 1:
        return NormalizationError(
            kind="not_single_statement", message=f"expected exactly one statement, got {len(parsed)}"
        )
    try:
        expression = normalize_identifiers(parsed[0], dialect=dialect)
        expression = normalize(expression)
        return simplify(expression, dialect=dialect)
    except SqlglotError as error:  # pragma: no cover - defensive: these passes don't raise on parseable input
        return NormalizationError(kind="normalize_failed", message=f"could not normalize ({error})", cause=error)


class ExecutionEquivalence:
    """Decides equivalence by running both queries and diffing their result sets.

    Delegates to `ResultSetEquivalence`, so the comparison runs in-platform with the case's
    `ComparisonConfig` (row order, NULL, float tolerance). Matching result sets yield
    `"equivalent"`, a computed difference yields `"not_equivalent"` (carrying the diff), and a
    query/budget failure yields `"unknown"`. Unlike a structural check this can refute, but on
    a single dataset a coincidental match is its own (accepted) limit.
    """

    name: SemanticEquivalenceMethod = "execution"

    def judge(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> SemanticVerdict:
        """Diff the model result against the gold query's result set in-platform.

        Args:
            case: The eval case, carrying the gold query and comparison config.
            output: The solver output, forwarded to `ResultSetEquivalence`.
            result: The executed model result, diffed against the gold query.
            context: The score context, carrying the budget-aware query runner.

        Returns:
            `"not_equivalent"` (with `diff`) when the result sets differ, `"equivalent"` when
            they match, or `"unknown"` when the comparison could not run.
        """
        score = ResultSetEquivalence().score(case, output, result, context=context)
        if score.diff is not None:
            return SemanticVerdict(method="execution", equivalence="not_equivalent", diff=score.diff)
        if score.passed:
            return SemanticVerdict(method="execution", equivalence="equivalent")
        return SemanticVerdict(method="execution", equivalence="unknown", detail=score.explanation)


def default_equivalence_checks() -> list[EquivalenceCheck]:
    """The default checks, cheapest and most portable first.

    Returns:
        A fresh list of the default `EquivalenceCheck`s: `AstEquivalence` then
        `ExecutionEquivalence`.
    """
    return [AstEquivalence(), ExecutionEquivalence()]


class SemanticEquivalence:
    """Scores a gold-query case by running equivalence checks until one decides."""

    def __init__(self, checks: Sequence[EquivalenceCheck] | None = None) -> None:
        """Bind the scorer to an ordered list of checks.

        Args:
            checks: The checks to run, in priority order; the first decisive verdict wins.
                Defaults to `default_equivalence_checks()` when omitted.
        """
        self._checks = list(checks) if checks is not None else default_equivalence_checks()

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Run the checks in order, stopping at the first decisive verdict.

        Args:
            case: The eval case; `expected` must be a `GoldQuery` (query-vs-query comparison).
            output: The solver output, passed through to each check.
            result: The executed model result, passed through to each check.
            context: The score context, carrying the model SQL, dialect, and query runner.

        Returns:
            A `ScoreResult` reflecting the first decisive verdict, or a failing result when no
            check could decide.

        Raises:
            TypeError: If `case.expected` is not a `GoldQuery`.
        """
        if not isinstance(case.expected, GoldQuery):
            msg = f"SemanticEquivalence requires a GoldQuery; got {type(case.expected).__name__}"
            raise TypeError(msg)
        verdicts: list[SemanticVerdict] = []
        for check in self._checks:
            verdict = check.judge(case, output, result, context=context)
            verdicts.append(verdict)
            if verdict.equivalence != "unknown":
                break
        return combine(verdicts, scorer=SCORER_NAME)
