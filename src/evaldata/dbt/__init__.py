"""dbt integration: load a dbt project's artifacts and evaluate its SQL and Semantic Layer.

`DbtContext` reads a built dbt `target/` directory (manifest.json, optional catalog.json, and
optional semantic_manifest.json) and exposes models, sources, schema context, and the semantic
layer's metrics and dimensions. `load_dbt` converts them into SQL eval cases and `load_dbt_metrics`
into Semantic Layer cases; `platform_from_profile` resolves the project's warehouse connection from
a dbt profile. `MetricCase`, `MetricLayerSolver`, `MetricSpecEquivalence`,
`MetricResultEquivalence`, and `MetricFirstDecisive` form the Semantic Layer evaluation vertical;
`evaluate_metric_case`, `assert_metric_eval`, and `run_metric_benchmark` are its runner surfaces.
"""

from evaldata.dbt.combinators import MetricFirstDecisive
from evaldata.dbt.context import (
    Column,
    DbtContext,
    DbtTest,
    Dimension,
    Entity,
    Measure,
    Metric,
    ModelRef,
    Relation,
    SchemaContext,
    SemanticLayerContext,
    SemanticModel,
    SourceRef,
    TableSchema,
)
from evaldata.dbt.errors import DbtError
from evaldata.dbt.eval import assert_metric_eval, evaluate_metric_case, run_metric_benchmark
from evaldata.dbt.loader import Mode, load_dbt, load_dbt_metrics
from evaldata.dbt.metric_layer_solver import SL_PROMPT_TEMPLATE, MetricLayerSolver
from evaldata.dbt.metric_result_equivalence import MetricResultEquivalence
from evaldata.dbt.metric_spec_equivalence import MetricSpecEquivalence
from evaldata.dbt.metricflow import CanonicalMetricQuery, canonicalize, run
from evaldata.dbt.profile import platform_from_profile
from evaldata.dbt.semantic_layer import (
    MetricCase,
    MetricQuery,
    MetricScorer,
    MetricSolver,
    MetricSolverOutput,
)

__all__ = [
    "SL_PROMPT_TEMPLATE",
    "CanonicalMetricQuery",
    "Column",
    "DbtContext",
    "DbtError",
    "DbtTest",
    "Dimension",
    "Entity",
    "Measure",
    "Metric",
    "MetricCase",
    "MetricFirstDecisive",
    "MetricLayerSolver",
    "MetricQuery",
    "MetricResultEquivalence",
    "MetricScorer",
    "MetricSolver",
    "MetricSolverOutput",
    "MetricSpecEquivalence",
    "Mode",
    "ModelRef",
    "Relation",
    "SchemaContext",
    "SemanticLayerContext",
    "SemanticModel",
    "SourceRef",
    "TableSchema",
    "assert_metric_eval",
    "canonicalize",
    "evaluate_metric_case",
    "load_dbt",
    "load_dbt_metrics",
    "platform_from_profile",
    "run",
    "run_metric_benchmark",
]
