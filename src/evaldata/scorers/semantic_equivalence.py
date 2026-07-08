"""`SemanticEquivalence`: query-vs-query equivalence via a sequence of pluggable checks.

Each `EquivalenceCheck` returns a `SemanticVerdict`; every check compares the queries
themselves rather than running them, so a check confirms equivalence (`"equivalent"`) or
returns `"unknown"`; it never refutes. `AstEquivalence` normalizes both queries' syntax
trees with SQLGlot and compares them: portable and execution-free.
"""

import functools
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.normalize import normalize
from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
from sqlglot.optimizer.simplify import Simplifier, gen

from evaldata.equivalence.semantic import combine
from evaldata.scorers.base import misconfigured
from evaldata.scorers.context import ScoreContext
from evaldata.scorers.sql import Dialect
from evaldata.types import (
    EquivalenceMethod,
    EvalCase,
    ExecutionResult,
    GoldQuery,
    NormalizationError,
    ScoreResult,
    SemanticVerdict,
    SolverOutput,
    Sql,
)

SCORER_NAME = "semantic_equivalence"


@runtime_checkable
class EquivalenceCheck(Protocol):
    """One way of confirming equivalence by comparing the queries, not their results; never raises.

    A check confirms equivalence (`"equivalent"`) or returns `"unknown"`; it never refutes, so
    it cannot falsely reject a correct query.
    """

    method: EquivalenceMethod

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
    failure, or input that is not exactly one statement) yields `"unknown"`, never a
    refutation. The normalization is schema-free: it fully reassociates commutative
    arithmetic (`+`/`*`), boolean/bitwise chains, and `IN`-list order; other unhandled
    equivalences fall through as `"unknown"`.
    """

    method: EquivalenceMethod = "ast"

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
    """Return an `"unknown"` AST verdict carrying `detail`."""
    return SemanticVerdict(method="ast", equivalence="unknown", detail=detail)


# Float `+`/`*` reassociation can differ in the last ULP; accepted, since this governs
# equivalence, not exact bit-identity.
_ASSOCIATIVE_COMMUTATIVE = (exp.And, exp.Or, exp.BitwiseAnd, exp.BitwiseOr, exp.BitwiseXor, exp.Add, exp.Mul)
_NONDETERMINISTIC = (
    exp.Rand,
    exp.Uuid,
    exp.CurrentTimestamp,
    exp.CurrentDate,
    exp.CurrentTime,
    exp.CurrentDatetime,
    exp.CurrentUser,
)
# Non-deterministic builtins sqlglot has no dedicated class for, so they are matched by name.
_NONDETERMINISTIC_NAMES = frozenset({"monotonically_increasing_id", "spark_partition_id", "input_file_name"})


def _is_non_deterministic(tree: exp.Expression) -> bool:
    """Whether `tree` contains a node whose value is not a function of its inputs.

    Args:
        tree: The parsed expression to scan.

    Returns:
        `True` if any node is non-deterministic.
    """
    for node in tree.walk():
        if isinstance(node, _NONDETERMINISTIC):
            return True
        if isinstance(node, (exp.Func, exp.Anonymous)):
            name = (node.name or node.sql_name()).lower()
            if name in _NONDETERMINISTIC_NAMES:
                return True
    return False


def _canonicalize(node: exp.Expression) -> exp.Expression:
    """Rewrite associative-commutative chains and `IN`-lists into a canonical order.

    Args:
        node: The expression to canonicalize; mutated in place and returned.

    Returns:
        The canonicalized expression (the same object as `node`, except for reassociated
        chains, which are rebuilt).
    """
    for key, value in list(node.args.items()):
        if isinstance(value, exp.Expression):
            node.set(key, _canonicalize(value))
        elif isinstance(value, list):
            node.set(key, [_canonicalize(v) if isinstance(v, exp.Expression) else v for v in value])
    if isinstance(node, _ASSOCIATIVE_COMMUTATIVE):
        parts = sorted(node.flatten(), key=gen)
        cls = type(node)
        return functools.reduce(lambda left, right: cls(this=left, expression=right), parts)
    if isinstance(node, exp.In) and len(node.expressions) > 1:
        node.set("expressions", sorted(node.expressions, key=gen))
    return node


def _normalize(sql: Sql, dialect: Dialect) -> exp.Expression | NormalizationError:
    """Parse and normalize `sql` into a comparable expression, or return a `NormalizationError`.

    Returns a `NormalizationError` (rather than raising) when `sql` does not parse, is not a
    single statement, contains a non-deterministic call, or hits an unfoldable constant (e.g.
    division by zero) while simplifying.

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
    if _is_non_deterministic(parsed[0]):
        return NormalizationError(
            kind="non_deterministic", message="query contains a non-deterministic call; cannot compare on syntax"
        )
    try:
        return _normalize_tree(parsed[0], dialect)
    except (SqlglotError, ArithmeticError) as error:
        return NormalizationError(kind="normalize_failed", message=f"could not normalize ({error})", cause=error)


class _PatchedSimplifier(Simplifier):
    """SQLGlot's simplifier with `simplify_equality` corrected for `constant - variable`.

    Upstream mis-folds these (`0 - a = 1` becomes `a = 1`, not `a = -1`), which would over-merge.
    Temporary until the fix ships in a SQLGlot release; a canary test fails once it lands, flagging
    removal.
    """

    def simplify_equality(self, expression: exp.Expression) -> exp.Expression:
        """Rewrite the `constant - variable` comparison correctly; defer the rest to SQLGlot.

        Args:
            expression: The comparison node.

        Returns:
            The corrected comparison when the variable is a subtraction's subtrahend, else
            SQLGlot's own result.
        """
        left = expression.left if isinstance(expression, self.COMPARISONS) else None
        if (
            isinstance(left, exp.Sub)
            and left.left.is_number
            and not left.right.is_number
            and expression.right.is_number
        ):
            comparison = self.INVERSE_COMPARISONS.get(type(expression), type(expression))
            return comparison(this=left.right, expression=exp.Sub(this=left.left, expression=expression.right))
        return super().simplify_equality(expression)


def _simplify(expression: exp.Expression, dialect: Dialect) -> exp.Expression:
    """Simplify `expression` with the local `constant - variable` correction applied.

    Args:
        expression: The expression to simplify.
        dialect: The SQLGlot dialect to simplify in.

    Returns:
        The simplified expression.
    """
    return _PatchedSimplifier(dialect=dialect).simplify(expression)


def _normalize_tree(tree: exp.Expression, dialect: Dialect) -> exp.Expression:
    """Normalize a parsed tree so that queries with equal normalized trees are equivalent.

    The input is not mutated.

    Args:
        tree: The parsed expression to normalize.
        dialect: The SQLGlot dialect to normalize in.

    Returns:
        The normalized expression.
    """
    expression = normalize_identifiers(tree.copy(), dialect=dialect)
    expression = _simplify(expression, dialect)
    expression = normalize(expression)
    expression = _canonicalize(expression.copy())
    expression = _simplify(expression, dialect)
    return _canonicalize(expression)  # second pass converges to a fixpoint


def default_equivalence_checks() -> list[EquivalenceCheck]:
    """The default checks, cheapest and most portable first.

    Returns:
        A fresh list of the default `EquivalenceCheck`s: just `AstEquivalence`.
    """
    return [AstEquivalence()]


class SemanticEquivalence:
    """Scores a gold-query case with checks that compare the queries themselves.

    It never runs a query and never refutes, so it confirms equivalence or is undecided. The
    first check that confirms yields a passing result; if none confirm, the result is
    inconclusive.
    """

    def __init__(self, checks: Sequence[EquivalenceCheck] | None = None) -> None:
        """Bind the scorer to an ordered list of checks.

        Args:
            checks: The checks to run, in priority order; the first that confirms wins.
                Defaults to `default_equivalence_checks()` when omitted.
        """
        self._checks = list(checks) if checks is not None else default_equivalence_checks()

    def score(
        self, case: EvalCase, output: SolverOutput, result: ExecutionResult, *, context: ScoreContext
    ) -> ScoreResult:
        """Run the checks in order, stopping at the first that confirms equivalence.

        Args:
            case: The eval case; `expected` must be a `GoldQuery` (query-vs-query comparison).
            output: The solver output, passed through to each check.
            result: The executed model result, passed through to each check.
            context: The score context, carrying the model SQL, dialect, and query runner.

        Returns:
            A passing `ScoreResult` when a check confirms equivalence, else an inconclusive
            result (no check could confirm, or `case.expected` is not a `GoldQuery`).
        """
        if not isinstance(case.expected, GoldQuery):
            return misconfigured(SCORER_NAME, case.expected, "a GoldQuery")
        verdicts: list[SemanticVerdict] = []
        for check in self._checks:
            verdict = check.judge(case, output, result, context=context)
            verdicts.append(verdict)
            if verdict.equivalence == "equivalent":
                break
        return combine(verdicts, scorer=SCORER_NAME)
