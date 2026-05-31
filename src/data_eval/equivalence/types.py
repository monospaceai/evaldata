"""Semantic SQL-type comparison via SQLGlot (single-dialect)."""

from sqlglot import exp
from sqlglot.errors import SqlglotError

from data_eval.types import SQLDialect


def types_match(actual: str, expected: str, dialect: SQLDialect) -> bool:
    """Check whether two SQL type strings are semantically equivalent in a dialect.

    Both strings are parsed with SQLGlot and compared by base type (scalars) or
    structurally (parameterized types), so aliases like `BIGINT`/`INT8` match. Falls back
    to literal string equality if either string fails to parse — graceful handling of
    exotic native types SQLGlot doesn't recognise, rather than crashing.

    Args:
        actual: The actual SQL type string.
        expected: The expected SQL type string.
        dialect: The SQLGlot dialect to parse both strings in.

    Returns:
        `True` if the two types are semantically equivalent in `dialect`.
    """
    try:
        actual_dt = exp.DataType.build(actual, dialect=dialect)
        expected_dt = exp.DataType.build(expected, dialect=dialect)
    except SqlglotError:
        return actual == expected
    return actual_dt.is_type(expected_dt)
