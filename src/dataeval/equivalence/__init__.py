"""Result-set equivalence engine: column reconciliation plus the pure `build_result_set_diff` assembly seam."""

from dataeval.equivalence.columns import ColumnReconciliation, reconcile_columns
from dataeval.equivalence.compare import build_result_set_diff

__all__ = [
    "ColumnReconciliation",
    "build_result_set_diff",
    "reconcile_columns",
]
