"""Tests for the `evaldata` CLI (`run` and `doctor`)."""

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import get_args

import pytest
from typer.testing import CliRunner

import evaldata.cli as cli
from evaldata.cli import _build_refs, app
from evaldata.types import PlatformKind

runner = CliRunner()


@pytest.mark.unit
class TestRun:
    def test_forwards_path_json_and_extra_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, list[str]] = {}

        def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        result = runner.invoke(app, ["run", "tests/unit", "--json", "out.json", "-k", "foo", "-x"])

        assert result.exit_code == 0
        cmd = captured["cmd"]
        assert cmd[:3] == [sys.executable, "-m", "pytest"]
        assert "tests/unit" in cmd
        assert "--evaldata-json=out.json" in cmd
        # Unknown pytest args pass straight through, in order.
        assert cmd[-2:] == ["-k", "foo"] or cmd[-3:] == ["-k", "foo", "-x"]
        assert "-x" in cmd

    def test_omits_path_and_json_when_not_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured: dict[str, list[str]] = {}

        def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
            captured["cmd"] = cmd
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        result = runner.invoke(app, ["run"])

        assert result.exit_code == 0
        assert captured["cmd"] == [sys.executable, "-m", "pytest"]

    def test_propagates_pytest_exit_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
            return subprocess.CompletedProcess(cmd, 1)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)
        result = runner.invoke(app, ["run"])

        assert result.exit_code == 1

    def test_executes_real_pytest_and_writes_artifact(self, tmp_path: Path) -> None:
        test_file = tmp_path / "test_generated.py"
        test_file.write_text(textwrap.dedent(_EVAL_TEST).replace("__DIR__", repr(str(tmp_path))))
        artifact = tmp_path / "results.json"

        result = runner.invoke(app, ["run", str(test_file), "--json", str(artifact)])

        assert result.exit_code == 0
        assert artifact.exists()


@pytest.mark.unit
class TestDoctor:
    @pytest.fixture(autouse=True)
    def _isolate_platform_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "EVALDATA_DUCKDB_PATH",
            "EVALDATA_POSTGRES_CONNINFO",
            "DATABRICKS_SERVER_HOSTNAME",
            "DATABRICKS_HTTP_PATH",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_ok_for_reachable_duckdb(self) -> None:
        result = runner.invoke(app, ["doctor", "--duckdb", ":memory:"])
        assert result.exit_code == 0
        assert "duckdb" in result.output
        assert "OK" in result.output

    def test_fail_for_unreachable_duckdb(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", "--duckdb", str(tmp_path / "nope" / "x.db")])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_fail_when_probe_query_returns_error_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A connection that opens but whose SELECT 1 fails as an ExecutionResult.error is a FAIL.
        from evaldata.types import ExecutionError, ExecutionResult

        class _ErroringAdapter:
            def execute(self, sql: str) -> ExecutionResult:
                return ExecutionResult(
                    rows=[],
                    schema=None,
                    latency_seconds=0.0,
                    error=ExecutionError(kind="query_failed", message="probe blew up"),
                )

            def close(self) -> None: ...

        monkeypatch.setattr(cli, "resolve", lambda ref: _ErroringAdapter())
        result = runner.invoke(app, ["doctor", "--duckdb", ":memory:"])
        assert result.exit_code == 1
        assert "FAIL" in result.output
        assert "probe blew up" in result.output

    def test_requires_at_least_one_platform(self) -> None:
        result = runner.invoke(app, ["doctor"])
        assert result.exit_code == 2
        assert "specify at least one platform" in result.output

    def test_covers_every_supported_kind(self) -> None:
        # Builds refs only — no connection attempted.
        refs = _build_refs(
            duckdb=":memory:",
            postgres="",
            databricks_server_hostname="h",
            databricks_http_path="/sql/1.0/warehouses/abc",
        )
        assert {ref.kind for ref in refs} == set(get_args(PlatformKind))

    def test_databricks_flags_required_together(self) -> None:
        result = runner.invoke(app, ["doctor", "--databricks-server-hostname", "h"])
        assert result.exit_code == 2
        assert "together" in result.output


_EVAL_TEST = """
    from pathlib import Path

    import duckdb

    from evaldata import CallableSolver, ResultSetEquivalence, assert_eval, eval_case
    from evaldata.platforms import duckdb_platform

    _DB = Path(__DIR__) / "t.duckdb"
    _con = duckdb.connect(str(_DB))
    _con.execute("CREATE TABLE t (genre VARCHAR)")
    _con.execute("INSERT INTO t VALUES ('Rock'), ('Rock')")
    _con.close()

    duck = duckdb_platform(name="p", path=str(_DB))

    @eval_case(input="q", expected={"rows": [{"c": 2}]}, platform=duck)
    def test_eval(case):
        solver = CallableSolver(lambda c: "SELECT count(*) AS c FROM t WHERE genre = 'Rock'")
        assert_eval(case, solver, scorers=[ResultSetEquivalence()])
"""
