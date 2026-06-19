"""The `dataeval` command-line interface."""

import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from dataeval.platforms.registry import (
    close_all,
    databricks_platform,
    duckdb_platform,
    postgres_platform,
    resolve,
)
from dataeval.types import PlatformRef

app = typer.Typer(help="AI evals for data & analytics engineering teams.", no_args_is_help=True)


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    path: str | None = typer.Argument(None, help="Path or test id to run; omit to use pytest's testpaths."),
    json_path: Path | None = typer.Option(
        None,
        "--json",
        metavar="PATH",
        help="Also write the structured dataeval results JSON to PATH (off by default).",
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
        cmd.append(f"--dataeval-json={json_path}")
    cmd.extend(ctx.args)
    completed = subprocess.run(cmd)  # noqa: PLW1510 - exit code is forwarded, not raised on
    raise typer.Exit(completed.returncode)


def _build_refs(
    *,
    duckdb: str | None,
    postgres: str | None,
    databricks_server_hostname: str | None = None,
    databricks_http_path: str | None = None,
) -> list[PlatformRef]:
    """Build a `PlatformRef` for each platform flag that was provided.

    Each branch routes through the typed registry builder, so a flag can only ever name a
    real `PlatformKind`. The Databricks ref is built only when both its server hostname and
    HTTP path are given (it has no single-value form).

    Args:
        duckdb: A DuckDB database path, or `None` if the flag was not given.
        postgres: A PostgreSQL conninfo, or `None` if the flag was not given.
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
        return False, result.error
    return True, "connected"


@app.command()
def doctor(
    duckdb: str | None = typer.Option(
        None, "--duckdb", metavar="PATH", envvar="DATAEVAL_DUCKDB_PATH", help="DuckDB database path to check."
    ),
    postgres: str | None = typer.Option(
        None,
        "--postgres",
        metavar="CONNINFO",
        envvar="DATAEVAL_POSTGRES_CONNINFO",
        help='PostgreSQL libpq conninfo to check (empty "" uses PG* env vars / libpq defaults).',
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
) -> None:
    """Check that the given platform connections work (one --<kind> flag per platform).

    Args:
        duckdb: A DuckDB database path to check (also read from `DATAEVAL_DUCKDB_PATH`).
        postgres: A PostgreSQL conninfo to check (also read from
            `DATAEVAL_POSTGRES_CONNINFO`).
        databricks_server_hostname: A Databricks workspace hostname to check (also read from
            `DATABRICKS_SERVER_HOSTNAME`); required together with `databricks_http_path`.
        databricks_http_path: A Databricks SQL Warehouse HTTP path to check (also read from
            `DATABRICKS_HTTP_PATH`); required together with `databricks_server_hostname`.

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
        databricks_server_hostname=databricks_server_hostname,
        databricks_http_path=databricks_http_path,
    )
    if not refs:
        msg = "specify at least one platform, e.g. --duckdb PATH or --postgres CONNINFO"
        raise typer.BadParameter(msg)

    console = Console()
    table = Table(title="dataeval doctor", title_justify="left")
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

    console.print(table)
    if not all_ok:
        raise typer.Exit(1)
