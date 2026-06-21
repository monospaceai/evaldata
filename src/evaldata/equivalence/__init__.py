"""Result-set equivalence engine: column reconciliation plus the pure `build_result_set_diff` assembly seam."""

from evaldata.equivalence.columns import ColumnReconciliation, reconcile_columns
from evaldata.equivalence.compare import build_result_set_diff
from evaldata.equivalence.semantic import combine

__all__ = [
    "ColumnReconciliation",
    "build_result_set_diff",
    "combine",
    "reconcile_columns",
]
