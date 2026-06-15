"""SQL builders for expectation pushdown: wrap the model's query and emit check SQL.

Every builder renders dialect-correct SQL via SQLGlot, quoting user column names so a
column named `select` or `order` is safe, and aliasing the derived table as `t`.
"""

import math
from decimal import Decimal
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from dataeval.types import PlatformKind, Schema, Sql, SQLDialect

Dialect = SQLDialect | PlatformKind

SAMPLE_LIMIT = 20


def _subquery(model_sql: Sql, dialect: Dialect) -> exp.Subquery:
    """Parse `model_sql` and wrap it as a subquery aliased `t`.

    Args:
        model_sql: The model's SQL.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        A `Subquery` expression `(<model_sql>) AS t`.
    """
    parsed = sqlglot.parse_one(model_sql, dialect=dialect)
    return exp.Subquery(this=parsed, alias=exp.TableAlias(this=exp.to_identifier("t")))


# Alias for the gold/reference query when it is wrapped as a subquery; double-underscored so it
# cannot collide with the `t`/`l`/`r`/`d`/`j`/`m` aliases the other builders pin.
_GOLD_ALIAS = "__gold__"


def gold_schema_probe(gold_sql: Sql, dialect: Dialect) -> Sql:
    """Build `SELECT * FROM (<gold_sql>) AS __gold__ LIMIT 0` to discover the gold schema.

    The zero-row wrapper lets the engine report the gold query's column names and types
    without materialising any rows.

    Args:
        gold_sql: The gold/reference query's SQL.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        The schema-discovery SQL string.
    """
    parsed = sqlglot.parse_one(gold_sql, dialect=dialect)
    sub = exp.Subquery(this=parsed, alias=exp.TableAlias(this=exp.to_identifier(_GOLD_ALIAS)))
    query = exp.select("*").from_(sub).limit(0)
    return Sql(query.sql(dialect=dialect))


def wrap_model(model_sql: Sql, select: str, dialect: Dialect) -> Sql:
    """Build `SELECT <select> FROM (<model_sql>) AS t`, rendered for `dialect`.

    Args:
        model_sql: The model's SQL.
        select: The projection clause (e.g. `count(*)`).
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        The wrapped SQL string.
    """
    query = exp.select(select).from_(_subquery(model_sql, dialect))
    return Sql(query.sql(dialect=dialect))


def row_count(model_sql: Sql, dialect: Dialect) -> Sql:
    """Build `SELECT count(*) FROM (<model_sql>) AS t`.

    Args:
        model_sql: The model's SQL.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        The row-count SQL string.
    """
    return wrap_model(model_sql, "count(*)", dialect)


def not_null_count(model_sql: Sql, column: str, dialect: Dialect) -> Sql:
    """Build `SELECT count(*) FROM (<model_sql>) AS t WHERE <column> IS NULL`.

    Args:
        model_sql: The model's SQL.
        column: The column checked for NULLs.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        The not-null count SQL string.
    """
    query = (
        exp.select("count(*)")
        .from_(_subquery(model_sql, dialect))
        .where(exp.column(column, quoted=True).is_(exp.null()))
    )
    return Sql(query.sql(dialect=dialect))


def not_null_sample(model_sql: Sql, column: str, dialect: Dialect) -> Sql:
    """Build `SELECT * FROM (<model_sql>) AS t WHERE <column> IS NULL LIMIT 20`.

    Args:
        model_sql: The model's SQL.
        column: The column checked for NULLs.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        The not-null sample SQL string (up to 20 offending rows).
    """
    query = (
        exp.select("*")
        .from_(_subquery(model_sql, dialect))
        .where(exp.column(column, quoted=True).is_(exp.null()))
        .limit(SAMPLE_LIMIT)
    )
    return Sql(query.sql(dialect=dialect))


def _duplicates_query(model_sql: Sql, column: str, projection: list[str], dialect: Dialect) -> exp.Select:
    """Build the duplicated-values query: non-NULL values grouped, `HAVING count(*) > 1`.

    Args:
        model_sql: The model's SQL.
        column: The column checked for uniqueness.
        projection: The select clauses over the grouped rows.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        A `Select` expression over the duplicated keys.
    """
    col = exp.column(column, quoted=True)
    return (
        exp.select(*projection)
        .from_(_subquery(model_sql, dialect))
        .where(exp.Not(this=col.is_(exp.null())))
        .group_by(col)
        .having(exp.Count(this=exp.Star()) > 1)
    )


def unique_count(model_sql: Sql, column: str, dialect: Dialect) -> Sql:
    """Build a count of duplicated (non-NULL) values: distinct keys appearing more than once.

    Args:
        model_sql: The model's SQL.
        column: The column checked for uniqueness.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        The unique-violation count SQL string.
    """
    inner = _duplicates_query(model_sql, column, [exp.column(column, quoted=True).sql(dialect=dialect)], dialect)
    outer = exp.select("count(*)").from_(exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier("d"))))
    return Sql(outer.sql(dialect=dialect))


def unique_sample(model_sql: Sql, column: str, dialect: Dialect) -> Sql:
    """Build a sample of up to 20 duplicated values with their counts (column `n`).

    Args:
        model_sql: The model's SQL.
        column: The column checked for uniqueness.
        dialect: The SQLGlot dialect to parse and render in.

    Returns:
        The unique-violation sample SQL string.
    """
    col = exp.column(column, quoted=True).sql(dialect=dialect)
    query = _duplicates_query(model_sql, column, [col, "count(*) AS n"], dialect).limit(SAMPLE_LIMIT)
    return Sql(query.sql(dialect=dialect))


def round_scale(tol: float) -> int:
    """Derive a decimal scale from an absolute float tolerance for `ROUND` matching.

    Args:
        tol: The absolute float tolerance; must be positive.

    Returns:
        `max(0, round(-log10(tol)))`, the number of fractional digits to round to.
    """
    return max(0, round(-math.log10(tol)))


def _literal(value: Any) -> exp.Expression:
    """Render a Python cell value as a SQLGlot literal expression (`NULL` for `None`).

    Args:
        value: The cell value (`None`, `bool`, `int`/`float`/`Decimal`, `str`, …).

    Returns:
        A SQLGlot expression for the literal.
    """
    if value is None:
        return exp.null()
    if isinstance(value, bool):
        return exp.convert(value)
    if isinstance(value, (int, float, Decimal)):
        return exp.Literal.number(str(value))
    if isinstance(value, str):
        return exp.Literal.string(value)
    return exp.convert(value)


def is_numeric_type(raw: str, dialect: Dialect) -> bool:
    """Whether a SQL type string names a numeric type in `dialect`.

    Args:
        raw: The native SQL type string.
        dialect: The SQLGlot dialect to parse `raw` in.

    Returns:
        `True` if `raw` parses to a numeric type, `False` otherwise (including unparseable).
    """
    try:
        return exp.DataType.build(raw, dialect=dialect).this in exp.DataType.NUMERIC_TYPES
    except SqlglotError:
        return False


# A wide fixed-point cast applied before `ROUND` so the tolerance scale is not truncated.
_ROUND_CAST = "DECIMAL(38, 18)"


def _maybe_round(value: exp.Expression, round_it: bool, scale: int) -> exp.Expression:
    """Wrap `value` in `ROUND(CAST(value AS DECIMAL(38, 18)), scale)` when `round_it`.

    Args:
        value: The expression to wrap.
        round_it: Whether to apply rounding.
        scale: The decimal scale passed to `ROUND`.

    Returns:
        The wrapped expression, or `value` unchanged when `round_it` is false.
    """
    if not round_it:
        return value
    fixed = exp.Cast(this=value, to=exp.DataType.build(_ROUND_CAST))
    return exp.func("ROUND", fixed, exp.Literal.number(scale))


def expected_relation(
    rows: list[dict[str, Any]],
    schema: Schema | None,
    in_both: list[str],
    dialect: Dialect,
    round_scale: int | None,
) -> exp.Query:
    """Materialise authored expected rows as a typed inline relation over `in_both`.

    Each row becomes `SELECT CAST(<lit> AS <type>) AS <col>, …`, `UNION ALL`-joined. Types
    come from `schema` (matched by name, in `in_both` order); a `None` cell is
    `CAST(NULL AS <type>)`. When `round_scale` is not `None`, numeric columns are wrapped in
    `ROUND(…, round_scale)`. When `schema` is `None`, literals are emitted untyped
    (best-effort) and no rounding is applied — string-vs-number distinctions are then left to
    the engine's literal types.

    Args:
        rows: The authored expected rows, keyed by column name.
        schema: The expected schema supplying per-column types, or `None` for untyped.
        in_both: The columns to project, in expected order.
        dialect: The SGLGlot dialect to render in.
        round_scale: The `ROUND` scale for numeric columns, or `None` for no rounding.

    Returns:
        A SQLGlot query (`SELECT …` or a `UNION ALL` chain) yielding the expected relation.
        An empty `rows` yields a `SELECT … WHERE 1 = 0` typed empty relation.
    """
    types = dict(zip(schema.names, schema.types, strict=True)) if schema is not None else {}

    def project(row: dict[str, Any]) -> exp.Select:
        selections: list[exp.Expression] = []
        for col in in_both:
            lit = _literal(row.get(col))
            if col in types:
                raw = types[col].raw
                try:
                    cell: exp.Expression = exp.Cast(this=lit, to=exp.DataType.build(raw, dialect=dialect))
                except SqlglotError:
                    cell = lit
                numeric = is_numeric_type(raw, dialect)
            else:
                cell = lit
                numeric = False
            cell = _maybe_round(cell, round_scale is not None and numeric, round_scale or 0)
            selections.append(cell.as_(exp.to_identifier(col, quoted=True)))
        return exp.Select(expressions=selections)

    if not rows:
        empty = project({}).where(exp.condition("1 = 0"))
        return empty

    relation: exp.Query = project(rows[0])
    for row in rows[1:]:
        relation = exp.union(relation, project(row), distinct=False)
    return relation


def _aligned_relation(
    base: exp.Subquery,
    in_both: list[str],
    numeric_columns: set[str],
    round_scale: int | None,
) -> exp.Select:
    """Project a base relation onto `in_both` (in order), optionally rounding numerics.

    Each column in `numeric_columns` is wrapped in `ROUND(<col>, round_scale)` when
    `round_scale` is not `None`; every column is quoted in both reference and alias.

    Args:
        base: The aliased subquery to project from (the model's or the gold query's relation).
        in_both: The columns to project, in expected order.
        numeric_columns: The subset of `in_both` to round when `round_scale` is set.
        round_scale: The `ROUND` scale for numeric columns, or `None` for no rounding.

    Returns:
        A `Select` projecting the aligned relation.
    """
    selections: list[exp.Expression] = []
    for col in in_both:
        column = exp.column(col, quoted=True)
        cell = _maybe_round(column, round_scale is not None and col in numeric_columns, round_scale or 0)
        selections.append(cell.as_(exp.to_identifier(col, quoted=True)))
    return exp.Select(expressions=selections).from_(base)


def aligned_actual(
    model_sql: Sql,
    in_both: list[str],
    numeric_columns: set[str],
    dialect: Dialect,
    round_scale: int | None,
) -> exp.Select:
    """Project the model's result onto `in_both` (in order), optionally rounding numerics.

    Builds `SELECT <cols> FROM (<model_sql>) AS t`, where each column in `numeric_columns`
    is wrapped in `ROUND(<col>, round_scale)` when `round_scale` is not `None`.

    Args:
        model_sql: The model's SQL.
        in_both: The columns to project, in expected order.
        numeric_columns: The subset of `in_both` to round when `round_scale` is set.
        dialect: The SQLGlot dialect to parse and render in.
        round_scale: The `ROUND` scale for numeric columns, or `None` for no rounding.

    Returns:
        A `Select` projecting the aligned actual relation.
    """
    return _aligned_relation(_subquery(model_sql, dialect), in_both, numeric_columns, round_scale)


def gold_expected(
    gold_sql: Sql,
    in_both: list[str],
    numeric_columns: set[str],
    dialect: Dialect,
    round_scale: int | None,
) -> exp.Select:
    """Project the gold query's result onto `in_both`, the expected side for a gold comparison.

    Builds `SELECT <cols> FROM (<gold_sql>) AS __gold__`, sharing the quoted-column projection
    and numeric `ROUND`/tolerance treatment with the actual side so the engine compares like
    for like.

    Args:
        gold_sql: The gold/reference query's SQL.
        in_both: The columns to project, in expected order.
        numeric_columns: The subset of `in_both` to round when `round_scale` is set.
        dialect: The SQLGlot dialect to parse and render in.
        round_scale: The `ROUND` scale for numeric columns, or `None` for no rounding.

    Returns:
        A `Select` projecting the aligned gold-expected relation.
    """
    parsed = sqlglot.parse_one(gold_sql, dialect=dialect)
    base = exp.Subquery(this=parsed, alias=exp.TableAlias(this=exp.to_identifier(_GOLD_ALIAS)))
    return _aligned_relation(base, in_both, numeric_columns, round_scale)


def _operand(relation: exp.Query, alias: str) -> exp.Select:
    """Wrap `relation` as `SELECT * FROM (<relation>) AS <alias>` to pin `EXCEPT ALL` precedence.

    Args:
        relation: The relation to wrap (a `SELECT` or a `UNION ALL` chain).
        alias: The subquery alias.

    Returns:
        A `Select` over the aliased subquery.
    """
    sub = exp.Subquery(this=relation.copy(), alias=exp.TableAlias(this=exp.to_identifier(alias)))
    return exp.select("*").from_(sub)


def _except_all(left: exp.Query, right: exp.Query) -> exp.Subquery:
    """Build the subquery `((left) EXCEPT ALL (right)) AS d`.

    Args:
        left: The left relation.
        right: The right relation.

    Returns:
        A `Subquery` over the bag difference, aliased `d`. Each operand is wrapped as a
        subquery so a `UNION ALL` operand associates correctly under `EXCEPT ALL`.
    """
    diff = exp.except_(_operand(left, "l"), _operand(right, "r"), distinct=False)
    return exp.Subquery(this=diff, alias=exp.TableAlias(this=exp.to_identifier("d")))


def except_all_count(left: exp.Query, right: exp.Query, dialect: Dialect) -> Sql:
    """Build `SELECT count(*) FROM ((left) EXCEPT ALL (right)) AS d`.

    Args:
        left: The left relation.
        right: The right relation.
        dialect: The SQLGlot dialect to render in.

    Returns:
        The bag-difference count SQL string.
    """
    query = exp.select("count(*)").from_(_except_all(left, right))
    return Sql(query.sql(dialect=dialect))


def except_all_sample(left: exp.Query, right: exp.Query, dialect: Dialect) -> Sql:
    """Build `SELECT * FROM ((left) EXCEPT ALL (right)) AS d LIMIT 20`.

    Args:
        left: The left relation.
        right: The right relation.
        dialect: The SQLGlot dialect to render in.

    Returns:
        The bag-difference sample SQL string (up to 20 rows).
    """
    query = exp.select("*").from_(_except_all(left, right)).limit(SAMPLE_LIMIT)
    return Sql(query.sql(dialect=dialect))


# Aliases pinning each side's columns through the join, so `e."order"`/`a."order"` survive as
# distinct `j."e__order"`/`j."a__order"` regardless of the user's column names.
_EXPECTED_PREFIX = "e__"
_ACTUAL_PREFIX = "a__"

# Per-side row-presence markers, projected as constant `TRUE` columns inside each operand;
# a `NULL` marker after the FULL OUTER JOIN means that side contributed no row for the key.
# Single-underscore names cannot collide with the double-underscore column prefixes, so a user
# column of any name (already renamed to `e__<col>`/`a__<col>`) is safe alongside them.
_EXPECTED_PRESENT = "e_present"
_ACTUAL_PRESENT = "a_present"


def _marked_operand(relation: exp.Query, prefix: str, marker: str, in_both: list[str], alias: str) -> exp.Subquery:
    """Wrap `relation`, renaming its columns to `<prefix><col>` and adding a `TRUE` marker.

    Renaming inside the operand (rather than `SELECT *`) keeps every user column under the
    double-underscore prefix, so the single-underscore presence marker can never collide with
    a user column of any name.

    Args:
        relation: The side relation (expected or actual), projecting the shared columns.
        prefix: The side prefix (`e__` or `a__`) applied to each shared column.
        marker: The constant-`TRUE` presence-marker column name.
        in_both: The shared columns to project, in expected order.
        alias: The subquery alias the join qualifies columns by.

    Returns:
        A `Subquery` exposing `<prefix><col>` for each shared column plus the presence marker.
    """
    inner = exp.Subquery(this=relation.copy(), alias=exp.TableAlias(this=exp.to_identifier("m")))
    selections: list[exp.Expression] = [
        exp.column(col, quoted=True).as_(exp.to_identifier(f"{prefix}{col}", quoted=True)) for col in in_both
    ]
    selections.append(exp.convert(True).as_(exp.to_identifier(marker)))
    marked = exp.select(*selections).from_(inner)
    return exp.Subquery(this=marked, alias=exp.TableAlias(this=exp.to_identifier(alias)))


def _keyed_join(
    expected_rel: exp.Query,
    actual_rel: exp.Query,
    key_columns: list[str],
    in_both: list[str],
    dialect: Dialect,
) -> exp.Subquery:
    """Build the `FULL OUTER JOIN` of expected and actual, aligned on `key_columns`.

    Each side renames its columns to `e__`/`a__`-prefixed names and carries a presence marker;
    the join condition is `e.<k> = a.<k>` ANDed over the key columns (collision-free and
    hash-joinable on both engines, unlike a null-safe operator which Postgres rejects in a
    `FULL JOIN`). A `NULL` in a key column never aligns, so such rows surface as missing/extra.

    Args:
        expected_rel: The expected relation over `in_both`.
        actual_rel: The actual relation over `in_both`.
        key_columns: The match-key columns aligned on.
        in_both: The shared columns, in expected order.
        dialect: The SQLGlot dialect to render in (unused for structure; kept for symmetry).

    Returns:
        A `Subquery` aliased `j` exposing `j."e__<col>"`, `j."a__<col>"`, `e_present`,
        and `a_present`.
    """
    expected = _marked_operand(expected_rel, _EXPECTED_PREFIX, _EXPECTED_PRESENT, in_both, "e")
    actual = _marked_operand(actual_rel, _ACTUAL_PREFIX, _ACTUAL_PRESENT, in_both, "a")
    condition = exp.and_(
        *(
            exp.EQ(
                this=exp.column(f"{_EXPECTED_PREFIX}{key}", table="e", quoted=True),
                expression=exp.column(f"{_ACTUAL_PREFIX}{key}", table="a", quoted=True),
            )
            for key in key_columns
        )
    )
    projections: list[exp.Expression] = []
    for col in in_both:
        projections.append(exp.column(f"{_EXPECTED_PREFIX}{col}", table="e", quoted=True))
        projections.append(exp.column(f"{_ACTUAL_PREFIX}{col}", table="a", quoted=True))
    projections.append(exp.column(_EXPECTED_PRESENT, table="e"))
    projections.append(exp.column(_ACTUAL_PRESENT, table="a"))
    joined = exp.select(*projections).from_(expected).join(actual, on=condition, join_type="full outer")
    return exp.Subquery(this=joined, alias=exp.TableAlias(this=exp.to_identifier("j")))


def _joined_column(prefix: str, col: str) -> exp.Column:
    """Reference a joined, prefixed column on the `j` subquery.

    Args:
        prefix: The side prefix (`e__` or `a__`).
        col: The original column name.

    Returns:
        The `j."<prefix><col>"` column expression.
    """
    return exp.column(f"{prefix}{col}", table="j", quoted=True)


def _absent(marker: str) -> exp.Expression:
    """Build `j.<marker> IS NULL` — the side contributed no row for the joined key.

    Args:
        marker: The presence-marker column name.

    Returns:
        The null-test expression.
    """
    return exp.column(marker, table="j").is_(exp.null())


def _both_present() -> exp.Expression:
    """Build `j.e_present IS NOT NULL AND j.a_present IS NOT NULL` — the key matched both sides.

    Returns:
        The both-present expression.
    """
    return exp.and_(
        exp.not_(exp.column(_EXPECTED_PRESENT, table="j").is_(exp.null())),
        exp.not_(exp.column(_ACTUAL_PRESENT, table="j").is_(exp.null())),
    )


def _tolerance_literal(tol: float) -> exp.Expression:
    """Build `CAST(<tol> AS DECIMAL(38, 18))` for an exact decimal tolerance band.

    Args:
        tol: The absolute tolerance.

    Returns:
        The cast tolerance expression.
    """
    return exp.cast(exp.Literal.number(repr(tol)), exp.DataType.build(_ROUND_CAST))


def _match_indicator(col: str, numeric: bool, null_equality: str, tol: exp.Expression) -> exp.Case:
    """Build a `0`/`1` `CASE` that is `1` iff the column matches between sides (never `NULL`).

    For `null_equality="equal"` two NULLs match; for `"distinct"` any NULL is a non-match.
    Numeric columns compare within the tolerance band `ABS(e - a) <= tol` over a fixed-point
    cast; other columns compare with `IS NOT DISTINCT FROM` (equal) or `=` (distinct).

    Args:
        col: The original column name.
        numeric: Whether the column is numeric (uses the tolerance band).
        null_equality: `"equal"` or `"distinct"`.
        tol: The cast tolerance expression.

    Returns:
        A `Case` expression evaluating to `1` (match) or `0` (mismatch).
    """
    distinct = null_equality == "distinct"
    one = exp.Literal.number(1)
    zero = exp.Literal.number(0)

    def either_null() -> exp.Expression:
        return exp.or_(
            _joined_column(_EXPECTED_PREFIX, col).is_(exp.null()), _joined_column(_ACTUAL_PREFIX, col).is_(exp.null())
        )

    if numeric:
        within = exp.LTE(
            this=exp.func(
                "ABS",
                exp.Sub(
                    this=exp.cast(_joined_column(_EXPECTED_PREFIX, col), exp.DataType.build(_ROUND_CAST)),
                    expression=exp.cast(_joined_column(_ACTUAL_PREFIX, col), exp.DataType.build(_ROUND_CAST)),
                ),
            ),
            expression=tol.copy(),
        )
        case = exp.Case()
        if distinct:
            case = case.when(either_null(), zero)
        else:
            both_null = exp.and_(
                _joined_column(_EXPECTED_PREFIX, col).is_(exp.null()),
                _joined_column(_ACTUAL_PREFIX, col).is_(exp.null()),
            )
            case = case.when(both_null, one).when(either_null(), zero)
        return case.when(within, one).else_(zero)

    if distinct:
        equal = exp.EQ(this=_joined_column(_EXPECTED_PREFIX, col), expression=_joined_column(_ACTUAL_PREFIX, col))
    else:
        equal = exp.NullSafeEQ(
            this=_joined_column(_EXPECTED_PREFIX, col), expression=_joined_column(_ACTUAL_PREFIX, col)
        )
    return exp.Case().when(equal, one).else_(zero)


def _sum_case(condition: exp.Expression, alias: str) -> exp.Expression:
    """Build `SUM(CASE WHEN <condition> THEN 1 ELSE 0 END) AS <alias>`.

    Args:
        condition: The predicate counted when true.
        alias: The output column alias.

    Returns:
        The aliased conditional-sum expression.
    """
    case = exp.Case().when(condition, exp.Literal.number(1)).else_(exp.Literal.number(0))
    return exp.func("SUM", case).as_(exp.to_identifier(alias))


def keyed_mismatch_alias(index: int) -> str:
    """The output alias for the `index`-th value column's mismatch count.

    Args:
        index: The value-column position.

    Returns:
        The alias `m<index>`.
    """
    return f"m{index}"


def keyed_diff_stats(
    expected_rel: exp.Query,
    actual_rel: exp.Query,
    key_columns: list[str],
    value_columns: list[str],
    numeric_columns: set[str],
    null_equality: str,
    tolerance: float,
    in_both: list[str],
    dialect: Dialect,
) -> Sql:
    """Build the one-row keyed-diff aggregate over the `FULL OUTER JOIN`.

    Projects `missing` (key only in expected), `extra` (key only in actual), and one
    `m<i>` per value column counting key-matched rows whose value differs.

    Args:
        expected_rel: The expected relation over `in_both`.
        actual_rel: The actual relation over `in_both`.
        key_columns: The match-key columns aligned on.
        value_columns: The non-key shared columns compared per row.
        numeric_columns: The subset of `value_columns` compared within the tolerance band.
        null_equality: `"equal"` or `"distinct"`.
        tolerance: The absolute tolerance for numeric columns.
        in_both: The shared columns, in expected order.
        dialect: The SQLGlot dialect to render in.

    Returns:
        The aggregate SQL string yielding one row of counts.
    """
    joined = _keyed_join(expected_rel, actual_rel, key_columns, in_both, dialect)
    tol = _tolerance_literal(tolerance)
    selects: list[exp.Expression] = [
        _sum_case(_absent(_ACTUAL_PRESENT), "missing"),
        _sum_case(_absent(_EXPECTED_PRESENT), "extra"),
    ]
    for index, col in enumerate(value_columns):
        indicator = _match_indicator(col, col in numeric_columns, null_equality, tol)
        differs = exp.and_(_both_present(), exp.EQ(this=indicator, expression=exp.Literal.number(0)))
        selects.append(_sum_case(differs, keyed_mismatch_alias(index)))
    query = exp.select(*selects).from_(joined)
    return Sql(query.sql(dialect=dialect))


def keyed_sample(
    expected_rel: exp.Query,
    actual_rel: exp.Query,
    key_columns: list[str],
    in_both: list[str],
    side: str,
    dialect: Dialect,
) -> Sql:
    """Build a bounded sample of one key-only bucket, projected back to original column names.

    Args:
        expected_rel: The expected relation over `in_both`.
        actual_rel: The actual relation over `in_both`.
        key_columns: The match-key columns aligned on.
        in_both: The shared columns, in expected order.
        side: `"missing"` (key only in expected) or `"extra"` (key only in actual).
        dialect: The SQLGlot dialect to render in.

    Returns:
        The sample SQL string (up to 20 rows) for the requested bucket.
    """
    joined = _keyed_join(expected_rel, actual_rel, key_columns, in_both, dialect)
    if side == "missing":
        prefix, absent_marker = _EXPECTED_PREFIX, _ACTUAL_PRESENT
    else:
        prefix, absent_marker = _ACTUAL_PREFIX, _EXPECTED_PRESENT
    selections = [_joined_column(prefix, col).as_(exp.to_identifier(col, quoted=True)) for col in in_both]
    query = exp.select(*selections).from_(joined).where(_absent(absent_marker)).limit(SAMPLE_LIMIT)
    return Sql(query.sql(dialect=dialect))


def keyed_dupes_count(relation: exp.Query, key_columns: list[str], dialect: Dialect) -> Sql:
    """Build a count of match-key values appearing more than once in `relation`.

    Used to reject a non-unique match key on either side; the engine defines key equality.
    `NULL` key values are excluded — they never align under the join's `=` condition, so two
    `NULL`-keyed rows are not a collision.

    Args:
        relation: The relation (expected or actual) over the shared columns.
        key_columns: The match-key columns.
        dialect: The SQLGlot dialect to render in.

    Returns:
        The duplicate-key count SQL string (`> 0` means the key is not unique).
    """
    sub = exp.Subquery(this=relation.copy(), alias=exp.TableAlias(this=exp.to_identifier("t")))
    not_null = exp.and_(*(exp.not_(exp.column(key, quoted=True).is_(exp.null())) for key in key_columns))
    inner = (
        exp.select(exp.Literal.number(1))
        .from_(sub)
        .where(not_null)
        .group_by(*(exp.column(key, quoted=True) for key in key_columns))
        .having(exp.Count(this=exp.Star()) > 1)
    )
    outer = exp.select("count(*)").from_(exp.Subquery(this=inner, alias=exp.TableAlias(this=exp.to_identifier("d"))))
    return Sql(outer.sql(dialect=dialect))
