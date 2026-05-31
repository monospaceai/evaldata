"""Result-set equivalence engine: a pure `compare()` that every result-comparing scorer wraps."""

from data_eval.equivalence.compare import compare
from data_eval.equivalence.result_set import TypedResultSet, UntypedResultSet

__all__ = ["TypedResultSet", "UntypedResultSet", "compare"]
