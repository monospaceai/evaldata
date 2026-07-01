"""`canonicalize` and `run`: resolve and execute a `MetricQuery` through MetricFlow.

Requires the optional `dbt-metricflow` toolchain (`dbt-sl` extra).
"""

import csv
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaldata.dbt.errors import DbtError
from evaldata.dbt.semantic_layer import MetricQuery

SpecKey = tuple[str, str, tuple[str, ...], str | None, str | None]


@dataclass(frozen=True)
class CanonicalMetricQuery:
    """A metric query resolved through MetricFlow: two queries with equal values are the same query."""

    metrics: frozenset[str]
    group_by: frozenset[SpecKey]
    order_by: tuple[tuple[SpecKey, bool], ...]
    limit: int | None
    where: frozenset[str]


def _spec_key(spec: Any) -> SpecKey:
    granularity = getattr(spec, "time_granularity", None)
    date_part = getattr(spec, "date_part", None)
    return (
        type(spec).__name__,
        spec.element_name,
        tuple(link.element_name for link in getattr(spec, "entity_links", ())),
        getattr(granularity, "name", None),
        date_part.value if date_part is not None else None,
    )


def canonicalize(query: MetricQuery, target_dir: str | Path) -> CanonicalMetricQuery | DbtError:
    """Resolve `query` against the project's semantic manifest into a comparable form.

    MetricFlow resolves default time grains and entity-linked paths, so semantically equal
    queries produce equal `CanonicalMetricQuery` values.

    Args:
        query: The metric query to resolve.
        target_dir: A dbt `target/` directory holding `semantic_manifest.json`.

    Returns:
        A `CanonicalMetricQuery`, or a `DbtError` if MetricFlow is not installed
        (`metricflow_unavailable`), the manifest is missing (`target_not_found`), or the query
        does not resolve (`metric_query_invalid`).
    """
    try:
        from metricflow_semantics.model.dbt_manifest_parser import parse_manifest_from_dbt_generated_manifest
        from metricflow_semantics.model.semantic_manifest_lookup import SemanticManifestLookup
        from metricflow_semantics.query.query_parser import MetricFlowQueryParser
    except ImportError as error:
        return DbtError(
            kind="metricflow_unavailable",
            message="dbt-metricflow is not installed; install the 'dbt-sl' extra to compare metric queries",
            cause=error,
        )

    manifest_path = Path(target_dir) / "semantic_manifest.json"
    if not manifest_path.is_file():
        return DbtError(kind="target_not_found", message=f"no semantic_manifest.json in {target_dir}")

    try:
        manifest = parse_manifest_from_dbt_generated_manifest(
            manifest_json_string=manifest_path.read_text(encoding="utf-8")
        )
        parser = MetricFlowQueryParser(SemanticManifestLookup(manifest))
        spec = parser.parse_and_validate_query(
            metric_names=query.metrics,
            group_by_names=query.group_by or None,
            where_constraint_strs=query.where or None,
            order_by_names=query.order_by or None,
            limit=query.limit,
        ).query_spec
    except Exception as error:  # MetricFlow raises a variety of parse and validation errors
        return DbtError(kind="metric_query_invalid", message=f"invalid metric query: {error}", cause=error)

    group_by = (*spec.dimension_specs, *spec.time_dimension_specs, *spec.entity_specs)
    return CanonicalMetricQuery(
        metrics=frozenset(s.element_name for s in spec.metric_specs),
        group_by=frozenset(_spec_key(s) for s in group_by),
        order_by=tuple((_spec_key(o.instance_spec), o.descending) for o in spec.order_by_specs),
        limit=spec.limit,
        where=frozenset(w.where_sql_template for w in spec.filter_intersection.where_filters),
    )


def _query_command(mf: str, query: MetricQuery, out_csv: Path) -> list[str]:
    command = [mf, "query", "--quiet", "--metrics", ",".join(query.metrics)]
    if query.group_by:
        command += ["--group-by", ",".join(query.group_by)]
    for predicate in query.where:
        command += ["--where", predicate]
    if query.order_by:
        command += ["--order", ",".join(query.order_by)]
    if query.limit is not None:
        command += ["--limit", str(query.limit)]
    return [*command, "--csv", str(out_csv)]


def run(
    query: MetricQuery, target_dir: str | Path, *, profiles_dir: str | Path | None = None
) -> list[dict[str, str]] | DbtError:
    """Execute `query` with the `mf` CLI against the project whose `target/` is `target_dir`.

    Args:
        query: The metric query to run.
        target_dir: A dbt `target/` directory; its parent is the project root `mf` runs in.
        profiles_dir: Where `mf` looks for `profiles.yml`; defaults to the project root.

    Returns:
        The result rows (column-to-value maps from the CSV export), or a `DbtError` if `mf`
        is not on PATH (`metricflow_unavailable`) or the query fails (`metric_query_invalid`).
    """
    mf = shutil.which("mf")
    if mf is None:
        return DbtError(
            kind="metricflow_unavailable", message="the 'mf' command is not on PATH; install the 'dbt-sl' extra"
        )
    project_dir = Path(target_dir).parent
    with tempfile.TemporaryDirectory() as tmp:
        out_csv = Path(tmp) / "result.csv"
        command = _query_command(mf, query, out_csv)
        env = {**os.environ, "DBT_PROFILES_DIR": str(profiles_dir if profiles_dir is not None else project_dir)}
        completed = subprocess.run(command, cwd=project_dir, env=env, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            return DbtError(kind="metric_query_invalid", message=f"mf query failed: {detail[:500]}")
        return list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
