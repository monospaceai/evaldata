"""The `evaldata` command-line interface."""

import enum
import json
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from evaldata.core import run_benchmark
from evaldata.core.runner import BenchmarkSummary
from evaldata.dbt import DbtError, load_dbt, platform_from_profile
from evaldata.loaders import load_bird, load_spider
from evaldata.loaders.benchmarks import SOURCES, cached_dataset_path, fetch_benchmark
from evaldata.platforms.registry import (
    close_all,
    databricks_platform,
    duckdb_platform,
    postgres_platform,
    resolve,
    sqlite_platform,
)
from evaldata.scorers import ExecutionAccuracy, ExpectationSuiteScorer, Scorer
from evaldata.solvers import SCHEMA_PROMPT_TEMPLATE, PromptSolver
from evaldata.types import EvalCase, PlatformRef

app = typer.Typer(help="AI evals for data & analytics engineering teams.", no_args_is_help=True)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    path: str | None = typer.Argument(None, help="Path or test id to run; omit to use pytest's testpaths."),
    json_path: Path | None = typer.Option(
        None,
        "--json",
        metavar="PATH",
        help="Also write the structured evaldata results JSON to PATH (off by default).",
    ),
) -> None:
    """Run the eval suite via pytest, forwarding any extra pytest arguments verbatim.

    Args:
        ctx: The Typer context; its extra args are forwarded straight to pytest.
        path: A path or test id to run; omit to use pytest's `testpaths`.
        json_path: If given, also write the structured results JSON to this path.

    Raises:
        Exit: Always, carrying pytest's return code as the process exit code.
    """
    cmd = [sys.executable, "-m", "pytest"]
    if path is not None:
        cmd.append(path)
    if json_path is not None:
        cmd.append(f"--evaldata-json={json_path}")
    cmd.extend(ctx.args)
    completed = subprocess.run(cmd)  # noqa: PLW1510 - exit code is forwarded, not raised on
    raise typer.Exit(completed.returncode)


class _Dataset(enum.StrEnum):
    """The benchmark datasets `bench` can run."""

    spider = "spider"
    bird = "bird"


_LOADERS: dict[_Dataset, Callable[..., Iterator[EvalCase]]] = {
    _Dataset.spider: load_spider,
    _Dataset.bird: load_bird,
}

_SCORERS: dict[_Dataset, Callable[[], Scorer]] = {
    _Dataset.spider: lambda: ExecutionAccuracy(column_alignment="by_value"),
    _Dataset.bird: lambda: ExecutionAccuracy(row_order="ignore", multiplicity="set"),
}


def _bench_stats(
    summary: BenchmarkSummary,
    difficulty_by_id: dict[str, str | None],
    *,
    dataset: _Dataset,
    model: str,
    split: str,
) -> dict[str, Any]:
    """Build the JSON stats payload for a benchmark run.

    Args:
        summary: The benchmark summary to serialize.
        difficulty_by_id: Maps each case id to its difficulty label, or `None` when the case
            carries none.
        dataset: The benchmark that was run.
        model: The litellm model id of the solver under test.
        split: The dataset split that was loaded.

    Returns:
        A JSON-serializable mapping with the run's identity, aggregate counts, a `by_difficulty`
        breakdown (empty when no case carries a difficulty), and every `CaseReport`.
    """
    by_difficulty: dict[str, dict[str, float | int]] = {}
    for report in summary.cases:
        difficulty = difficulty_by_id.get(report.id)
        if difficulty is None:
            continue
        bucket = by_difficulty.setdefault(difficulty, {"total": 0, "passed": 0, "accuracy": 0.0})
        bucket["total"] = int(bucket["total"]) + 1
        bucket["passed"] = int(bucket["passed"]) + (1 if report.passed else 0)
    for bucket in by_difficulty.values():
        total = int(bucket["total"])
        bucket["accuracy"] = int(bucket["passed"]) / total if total else 0.0
    return {
        "dataset": dataset.value,
        "model": model,
        "split": split,
        "total": summary.total,
        "passed": summary.passed,
        "accuracy": summary.accuracy,
        "by_difficulty": by_difficulty,
        "cases": [r.model_dump(mode="json") for r in summary.cases],
    }


@app.command()
def bench(
    dataset: _Dataset = typer.Argument(..., help="The benchmark to run."),
    path: Path | None = typer.Argument(
        None, help="Path to the unzipped dataset directory; omit to use the downloaded cache."
    ),
    model: str = typer.Option(..., "--model", help="litellm model id for the solver under test."),
    split: str = typer.Option("dev", "--split", help="Dataset split to load."),
    limit: int | None = typer.Option(None, "--limit", help="Run at most this many cases."),
    json_path: Path | None = typer.Option(
        None,
        "--json",
        metavar="PATH",
        help="Also write a JSON stats artifact (identity, counts, by-difficulty, cases) to PATH.",
    ),
) -> None:
    """Run a text-to-SQL benchmark and print its execution accuracy (EX).

    Loads the dataset's cases, runs a single-prompt LLM solver (with the schema injected into
    the prompt) against each, scores with the benchmark's `ExecutionAccuracy` configuration
    (Spider matches columns by value; BIRD uses order-insensitive set semantics), and prints the
    aggregate EX. When the cases carry a `difficulty`, also prints a per-difficulty breakdown.

    Args:
        dataset: The benchmark to run (`spider` or `bird`).
        path: Path to the unzipped dataset directory; omit to use the downloaded cache.
        model: A litellm model id for the solver under test.
        split: The dataset split to load (e.g. `dev`).
        limit: Run at most this many cases, or all of them when omitted.
        json_path: If given, also write a JSON stats artifact to this path.

    Raises:
        BadParameter: If `path` is omitted and the dataset has not been downloaded.
    """
    if path is None:
        path = cached_dataset_path(dataset.value)
        if path is None:
            msg = f"dataset not downloaded; run: evaldata fetch {dataset.value}"
            raise typer.BadParameter(msg)
    cases = list(_LOADERS[dataset](path, split=split))
    difficulty_by_id = {c.id: c.metadata.get("difficulty") for c in cases}
    solver = PromptSolver(model, prompt_template=SCHEMA_PROMPT_TEMPLATE, temperature=0)
    try:
        summary = run_benchmark(cases, solver, scorers=[_SCORERS[dataset]()], limit=limit)
    finally:
        close_all()  # this CLI invocation owns the per-db adapters it resolved

    console = Console()
    console.print(f"EX ({dataset.value}): {summary.accuracy:.1%} ({summary.passed}/{summary.total})")

    counts: dict[str, list[int]] = {}
    for report in summary.cases:
        difficulty = difficulty_by_id.get(report.id)
        if difficulty is None:
            continue
        bucket = counts.setdefault(difficulty, [0, 0])
        bucket[0] += 1 if report.passed else 0
        bucket[1] += 1
    if counts:
        table = Table(title=f"EX by difficulty ({dataset.value})", title_justify="left")
        table.add_column("difficulty")
        table.add_column("EX")
        table.add_column("passed/total")
        for difficulty in sorted(counts):
            passed, total = counts[difficulty]
            table.add_row(difficulty, f"{passed / total:.1%}", f"{passed}/{total}")
        console.print(table)

    if json_path is not None:
        stats = _bench_stats(summary, difficulty_by_id, dataset=dataset, model=model, split=split)
        json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


class _DbtMode(enum.StrEnum):
    """How `dbt-bench` builds cases."""

    authored = "authored"
    model = "model"
    tests = "tests"


@app.command(name="dbt-bench")
def dbt_bench(
    project_dir: Path = typer.Argument(..., help="dbt project directory (holds dbt_project.yml and profiles.yml)."),
    model: str = typer.Option(..., "--model", help="litellm model id for the solver under test."),
    cases_file: Path | None = typer.Option(None, "--cases", help="Cases YAML file (required for --mode authored)."),
    mode: _DbtMode = typer.Option(
        _DbtMode.authored, "--mode", help="Build cases from a cases file, documented models, or their data tests."
    ),
    target_dir: Path | None = typer.Option(
        None, "--target-dir", help="dbt artifacts directory; defaults to <project_dir>/target."
    ),
    profiles_dir: Path | None = typer.Option(
        None, "--profiles-dir", help="Directory holding profiles.yml; defaults to the project directory."
    ),
    target: str | None = typer.Option(
        None, "--target", help="dbt profile target name; defaults to the profile's target."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Run at most this many cases."),
    json_path: Path | None = typer.Option(
        None, "--json", metavar="PATH", help="Also write a JSON stats artifact to PATH."
    ),
) -> None:
    """Run a dbt project as a text-to-SQL benchmark and print its execution accuracy (EX).

    Resolves the project's warehouse from its dbt profile, builds cases (from a cases file, the
    documented models, or their data tests), runs a single-prompt LLM solver with the project's
    schema in the prompt, scores each case (execution accuracy, or the expectation suite in
    `tests` mode), and prints the aggregate pass rate.

    Args:
        project_dir: The dbt project directory (holding `dbt_project.yml`).
        model: A litellm model id for the solver under test.
        cases_file: Path to the cases YAML file (required for `authored` mode).
        mode: `authored` to read the cases file, `model` to derive cases from documented models,
            or `tests` to build expectation suites from documented models' data tests.
        target_dir: The dbt artifacts directory; defaults to `<project_dir>/target`.
        profiles_dir: Directory holding `profiles.yml`; defaults to `project_dir`.
        target: The dbt profile target name; defaults to the profile's `target`.
        limit: Run at most this many cases, or all of them when omitted.
        json_path: If given, also write a JSON stats artifact to this path.

    Raises:
        Exit: With code 1 if the profile or cases cannot be resolved.
    """
    console = Console()
    platform = platform_from_profile(project_dir, profiles_dir=profiles_dir, target=target)
    if isinstance(platform, DbtError):
        console.print(Text(platform.message, style="red"))
        raise typer.Exit(1)

    artifacts = target_dir if target_dir is not None else project_dir / "target"
    cases = load_dbt(artifacts, platform=platform, cases=cases_file, mode=mode.value)
    if isinstance(cases, DbtError):
        console.print(Text(cases.message, style="red"))
        raise typer.Exit(1)

    solver = PromptSolver(model, prompt_template=SCHEMA_PROMPT_TEMPLATE, temperature=0)
    scorer: Scorer = (
        ExpectationSuiteScorer()
        if mode is _DbtMode.tests
        else ExecutionAccuracy(row_order="ignore", multiplicity="set")
    )
    try:
        summary = run_benchmark(cases, solver, scorers=[scorer], limit=limit)
    finally:
        close_all()  # this CLI invocation owns the adapter it resolved

    console.print(f"EX (dbt): {summary.accuracy:.1%} ({summary.passed}/{summary.total})")
    if json_path is not None:
        stats = {
            "model": model,
            "mode": mode.value,
            "total": summary.total,
            "passed": summary.passed,
            "accuracy": summary.accuracy,
            "cases": [r.model_dump(mode="json") for r in summary.cases],
        }
        json_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


@app.command()
def fetch(
    dataset: str = typer.Argument(..., help="The benchmark dataset to download."),
    force: bool = typer.Option(False, "--force", help="Re-download even if a valid cached copy exists."),
    trust: bool = typer.Option(
        False, "--trust", help="Accept an unpinned source's bytes (required the first time it is fetched)."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", metavar="PATH", help="Cache directory to download into (else the default cache root)."
    ),
) -> None:
    """Download, verify, and cache a benchmark dataset for `bench` to read.

    Args:
        dataset: The benchmark dataset to download.
        force: Re-download even if a valid cached copy exists.
        trust: Accept an unpinned source's bytes (required the first time it is fetched).
        cache_dir: A cache directory to download into, else the default cache root.

    Raises:
        BadParameter: If `dataset` is not a known benchmark source.
        Exit: With code 1 if verification or validation fails.
    """
    if dataset not in SOURCES:
        available = ", ".join(sorted(SOURCES))
        msg = f"unknown dataset {dataset!r}; available: {available}"
        raise typer.BadParameter(msg)

    console = Console()
    try:
        root = fetch_benchmark(dataset, force=force, trust=trust, cache_dir=cache_dir)
    except RuntimeError as e:
        console.print(Text(str(e), style="red"))
        raise typer.Exit(1) from e
    console.print(f"cached at: {root}")


def _build_refs(
    *,
    duckdb: str | None,
    postgres: str | None,
    sqlite: str | None = None,
    databricks_server_hostname: str | None = None,
    databricks_http_path: str | None = None,
) -> list[PlatformRef]:
    """Build a `PlatformRef` for each platform flag that was provided.

    The Databricks ref requires both `databricks_server_hostname` and `databricks_http_path`;
    either alone produces no entry.

    Args:
        duckdb: A DuckDB database path, or `None` if the flag was not given.
        postgres: A PostgreSQL conninfo, or `None` if the flag was not given.
        sqlite: A SQLite database path, or `None` if the flag was not given.
        databricks_server_hostname: A Databricks workspace hostname, or `None`.
        databricks_http_path: A Databricks SQL Warehouse HTTP path, or `None`.

    Returns:
        One `PlatformRef` per platform whose flag(s) were provided, in flag order.
    """
    refs: list[PlatformRef] = []
    if duckdb is not None:
        refs.append(duckdb_platform(name="duckdb", path=duckdb))
    if postgres is not None:
        refs.append(postgres_platform(name="postgres", conninfo=postgres))
    if sqlite is not None:
        refs.append(sqlite_platform(name="sqlite", path=sqlite))
    if databricks_server_hostname is not None and databricks_http_path is not None:
        refs.append(
            databricks_platform(
                name="databricks",
                server_hostname=databricks_server_hostname,
                http_path=databricks_http_path,
            )
        )
    return refs


def _probe(ref: PlatformRef) -> tuple[bool, str]:
    """Resolve `ref` to a live adapter and run `SELECT 1`.

    Catches broadly on purpose: adapter construction can raise (e.g. psycopg fails to
    connect, or an optional driver is missing), and `doctor` must report that as a FAIL
    rather than crash. A query that fails as a value (`ExecutionResult.error`) is a FAIL
    too.

    Args:
        ref: The platform reference to probe.

    Returns:
        A tuple `(ok, detail)`: `ok` is whether the probe succeeded, and `detail` is a
        human-readable status or error message.
    """
    try:
        result = resolve(ref).execute("SELECT 1")
    except Exception as e:  # noqa: BLE001 - diagnostics: any failure is a reported FAIL
        return False, str(e)
    if result.error is not None:
        return False, result.error.message
    return True, "connected"


@app.command()
def doctor(
    duckdb: str | None = typer.Option(
        None, "--duckdb", metavar="PATH", envvar="EVALDATA_DUCKDB_PATH", help="DuckDB database path to check."
    ),
    postgres: str | None = typer.Option(
        None,
        "--postgres",
        metavar="CONNINFO",
        envvar="EVALDATA_POSTGRES_CONNINFO",
        help='PostgreSQL libpq conninfo to check (empty "" uses PG* env vars / libpq defaults).',
    ),
    sqlite: str | None = typer.Option(
        None, "--sqlite", metavar="PATH", envvar="EVALDATA_SQLITE_PATH", help="SQLite database path to check."
    ),
    databricks_server_hostname: str | None = typer.Option(
        None,
        "--databricks-server-hostname",
        metavar="HOST",
        envvar="DATABRICKS_SERVER_HOSTNAME",
        help="Databricks workspace hostname to check (paired with --databricks-http-path).",
    ),
    databricks_http_path: str | None = typer.Option(
        None,
        "--databricks-http-path",
        metavar="PATH",
        envvar="DATABRICKS_HTTP_PATH",
        help="Databricks SQL Warehouse HTTP path to check (paired with --databricks-server-hostname).",
    ),
    dbt_project: Path | None = typer.Option(
        None, "--dbt-project", metavar="DIR", help="dbt project directory to resolve via its profile and check."
    ),
) -> None:
    """Check that the given platform connections work (one --<kind> flag per platform).

    Args:
        duckdb: A DuckDB database path to check (also read from `EVALDATA_DUCKDB_PATH`).
        postgres: A PostgreSQL conninfo to check (also read from
            `EVALDATA_POSTGRES_CONNINFO`).
        sqlite: A SQLite database path to check (also read from `EVALDATA_SQLITE_PATH`).
        databricks_server_hostname: A Databricks workspace hostname to check (also read from
            `DATABRICKS_SERVER_HOSTNAME`); required together with `databricks_http_path`.
        databricks_http_path: A Databricks SQL Warehouse HTTP path to check (also read from
            `DATABRICKS_HTTP_PATH`); required together with `databricks_server_hostname`.
        dbt_project: A dbt project directory whose profile target is resolved to a platform and
            checked.

    Raises:
        BadParameter: If no platform flag is provided, or only one of the two Databricks flags is.
        Exit: With code 1 if any platform connection fails.
    """
    if (databricks_server_hostname is None) != (databricks_http_path is None):
        msg = "--databricks-server-hostname and --databricks-http-path must be given together"
        raise typer.BadParameter(msg)
    refs = _build_refs(
        duckdb=duckdb,
        postgres=postgres,
        sqlite=sqlite,
        databricks_server_hostname=databricks_server_hostname,
        databricks_http_path=databricks_http_path,
    )
    dbt_failure: DbtError | None = None
    if dbt_project is not None:
        resolved = platform_from_profile(dbt_project)
        if isinstance(resolved, DbtError):
            dbt_failure = resolved
        else:
            refs.append(resolved)
    if not refs and dbt_failure is None:
        msg = "specify at least one platform, e.g. --duckdb PATH or --dbt-project DIR"
        raise typer.BadParameter(msg)

    console = Console()
    table = Table(title="evaldata doctor", title_justify="left")
    table.add_column("platform")
    table.add_column("kind")
    table.add_column("status")

    all_ok = True
    try:
        for ref in refs:
            ok, detail = _probe(ref)
            all_ok = all_ok and ok
            mark = "OK" if ok else "FAIL"
            # Text (not markup) so bracketed driver messages render verbatim.
            table.add_row(ref.name, ref.kind, Text(f"{mark} {detail}", style="green" if ok else "red"))
    finally:
        close_all()  # this CLI invocation owns the adapters it resolved

    if dbt_failure is not None:
        all_ok = False
        table.add_row("dbt", "—", Text(f"FAIL {dbt_failure.message}", style="red"))

    console.print(table)
    if not all_ok:
        raise typer.Exit(1)
