"""Tests for the `evaldata` CLI commands."""

import json
import shutil
import sqlite3
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

FIXTURE_DBT = Path(__file__).parent.parent / "dbt" / "fixtures" / "jaffle_duckdb"


def _copy_dbt_project(tmp_path: Path) -> Path:
    dest = tmp_path / "jaffle"
    shutil.copytree(FIXTURE_DBT, dest, ignore=shutil.ignore_patterns("target", "logs", "dbt_packages"))
    return dest


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
class TestBench:
    def _make_spider(self, root: Path) -> None:
        db_dir = root / "database" / "clibench"
        db_dir.mkdir(parents=True)
        con = sqlite3.connect(db_dir / "clibench.sqlite")
        con.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        con.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "b")])
        con.commit()
        con.close()
        (root / "dev.json").write_text(
            json.dumps(
                [
                    {"db_id": "clibench", "question": "how many items?", "query": "SELECT count(*) FROM items"},
                    {"db_id": "clibench", "question": "names?", "query": "SELECT name FROM items"},
                ]
            )
        )

    def test_prints_execution_accuracy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        self._make_spider(tmp_path)
        # The mocked model always answers with the count query: it matches the first gold and
        # misses the second, so EX is 1/2.
        real_completion = litellm.completion
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: real_completion(**kwargs, mock_response="SELECT count(*) FROM items"),
        )

        result = runner.invoke(app, ["bench", "spider", str(tmp_path), "--model", "openai/gpt-4o-mini"])

        assert result.exit_code == 0
        assert "EX (spider): 50.0% (1/2)" in result.output

    def _make_bird(self, root: Path) -> None:
        db_dir = root / "dev_databases" / "clibench"
        db_dir.mkdir(parents=True)
        con = sqlite3.connect(db_dir / "clibench.sqlite")
        con.execute("CREATE TABLE items (id INTEGER, name TEXT)")
        # Two rows share a name, so set semantics dedups them to one.
        con.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a"), (2, "a"), (3, "b")])
        con.commit()
        con.close()
        (root / "dev.json").write_text(
            json.dumps(
                [
                    {
                        "db_id": "clibench",
                        "question": "distinct names?",
                        "evidence": "",
                        "SQL": "SELECT name FROM items",
                        "question_id": 0,
                        "difficulty": "simple",
                    },
                    {
                        "db_id": "clibench",
                        "question": "how many items?",
                        "evidence": "",
                        "SQL": "SELECT count(*) FROM items",
                        "question_id": 1,
                        "difficulty": "moderate",
                    },
                ]
            )
        )

    def test_bird_breakdown_and_json_artifact(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        self._make_bird(tmp_path)
        # The model answers with a deduping DISTINCT query. Under BIRD set semantics it matches
        # the first gold (which returns duplicate names), and misses the count, so EX is 1/2.
        real_completion = litellm.completion
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: real_completion(**kwargs, mock_response="SELECT DISTINCT name FROM items"),
        )
        artifact = tmp_path / "stats.json"

        result = runner.invoke(
            app,
            ["bench", "bird", str(tmp_path), "--model", "openai/gpt-4o-mini", "--json", str(artifact)],
        )

        assert result.exit_code == 0
        assert "EX (bird): 50.0% (1/2)" in result.output
        assert "simple" in result.output
        assert "moderate" in result.output

        assert artifact.exists()
        stats = json.loads(artifact.read_text())
        assert stats["dataset"] == "bird"
        assert stats["model"] == "openai/gpt-4o-mini"
        assert stats["total"] == 2
        assert stats["passed"] == 1
        assert stats["by_difficulty"] == {
            "simple": {"total": 1, "passed": 1, "accuracy": 1.0},
            "moderate": {"total": 1, "passed": 0, "accuracy": 0.0},
        }
        assert len(stats["cases"]) == 2

    def test_no_path_and_no_cache_guides_to_fetch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli, "cached_dataset_path", lambda name: None)
        result = runner.invoke(app, ["bench", "bird", "--model", "openai/gpt-4o-mini"])
        assert result.exit_code != 0
        assert "run: evaldata fetch bird" in result.output

    def test_no_path_resolves_from_cache(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        self._make_bird(tmp_path)
        monkeypatch.setattr(cli, "cached_dataset_path", lambda name: tmp_path)
        real_completion = litellm.completion
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: real_completion(**kwargs, mock_response="SELECT DISTINCT name FROM items"),
        )
        result = runner.invoke(app, ["bench", "bird", "--model", "openai/gpt-4o-mini"])
        assert result.exit_code == 0
        assert "EX (bird): 50.0% (1/2)" in result.output


@pytest.mark.unit
class TestBenchStats:
    """Unit tests for `_bench_stats`: covers the difficulty=None skip (line 110)."""

    def _make_report(self, id_: str, passed: bool) -> "object":
        from evaldata.reporting.collector import CaseReport

        return CaseReport(id=id_, input="q", passed=passed)

    def test_skips_cases_without_difficulty(self) -> None:
        from evaldata.cli import _bench_stats
        from evaldata.core.runner import BenchmarkSummary

        report = self._make_report("q1", True)
        summary = BenchmarkSummary(total=1, passed=1, accuracy=1.0, cases=[report])
        stats = _bench_stats(
            summary,
            {"q1": None},
            dataset=cli._Dataset.spider,
            model="m",
            split="dev",
        )
        assert stats["by_difficulty"] == {}

    def test_aggregates_difficulty_buckets(self) -> None:
        from evaldata.cli import _bench_stats
        from evaldata.core.runner import BenchmarkSummary

        reports = [self._make_report("q1", True), self._make_report("q2", False)]
        summary = BenchmarkSummary(total=2, passed=1, accuracy=0.5, cases=reports)
        stats = _bench_stats(
            summary,
            {"q1": "simple", "q2": "simple"},
            dataset=cli._Dataset.spider,
            model="m",
            split="dev",
        )
        assert stats["by_difficulty"]["simple"]["total"] == 2
        assert stats["by_difficulty"]["simple"]["passed"] == 1
        assert stats["by_difficulty"]["simple"]["accuracy"] == 0.5


@pytest.mark.unit
class TestDbtBench:
    def test_authored_prints_ex_and_writes_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        project = _copy_dbt_project(tmp_path)
        (project / "cases.yml").write_text(
            "- question: how many customers?\n  gold_sql: select count(*) as n from customers\n"
        )
        real_completion = litellm.completion
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: real_completion(**kwargs, mock_response="select count(*) as n from customers"),
        )
        artifact = tmp_path / "stats.json"

        result = runner.invoke(
            app,
            [
                "dbt-bench",
                str(project),
                "--model",
                "openai/gpt-4o-mini",
                "--cases",
                str(project / "cases.yml"),
                "--target-dir",
                str(project / "artifacts"),
                "--json",
                str(artifact),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "EX (dbt): 100.0% (1/1)" in result.output
        stats = json.loads(artifact.read_text())
        assert stats["mode"] == "authored"
        assert stats["total"] == 1
        assert stats["passed"] == 1

    def test_model_mode_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        project = _copy_dbt_project(tmp_path)
        real_completion = litellm.completion
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: real_completion(**kwargs, mock_response="select 1 as n"),
        )

        result = runner.invoke(
            app,
            [
                "dbt-bench",
                str(project),
                "--model",
                "openai/gpt-4o-mini",
                "--mode",
                "model",
                "--target-dir",
                str(project / "artifacts"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "EX (dbt): 0.0% (0/3)" in result.output

    def test_tests_mode_runs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import litellm

        project = _copy_dbt_project(tmp_path)
        # `customers` has unique + not_null tests on customer_id, which `select * from customers`
        # satisfies, so the one tests-mode case passes.
        real_completion = litellm.completion
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: real_completion(**kwargs, mock_response="select * from customers"),
        )

        result = runner.invoke(
            app,
            [
                "dbt-bench",
                str(project),
                "--model",
                "openai/gpt-4o-mini",
                "--mode",
                "tests",
                "--target-dir",
                str(project / "artifacts"),
            ],
        )

        assert result.exit_code == 0, result.output
        assert "EX (dbt): 100.0% (1/1)" in result.output

    def test_profile_error_exits_1(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app, ["dbt-bench", str(tmp_path), "--model", "openai/gpt-4o-mini", "--target-dir", str(tmp_path)]
        )
        assert result.exit_code == 1
        assert "dbt_project.yml" in result.output

    def test_missing_cases_exits_1(self, tmp_path: Path) -> None:
        project = _copy_dbt_project(tmp_path)
        result = runner.invoke(
            app,
            ["dbt-bench", str(project), "--model", "openai/gpt-4o-mini", "--target-dir", str(project / "artifacts")],
        )
        assert result.exit_code == 1
        assert "cases" in result.output


@pytest.mark.unit
class TestFetchCommand:
    def test_unknown_dataset_bad_parameter(self) -> None:
        result = runner.invoke(app, ["fetch", "notadataset"])
        assert result.exit_code == 2
        assert "unknown dataset" in result.output

    def test_success_prints_cached_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_root = tmp_path / "bird"
        fake_root.mkdir(parents=True)
        monkeypatch.setattr(cli, "fetch_benchmark", lambda *a, **kw: fake_root)
        result = runner.invoke(app, ["fetch", "bird"])
        assert result.exit_code == 0
        assert "cached at:" in result.output

    def test_runtime_error_exits_1_with_message(self, monkeypatch: pytest.MonkeyPatch) -> None:
        msg = "hash mismatch!"

        def _fail(*args: object, **kwargs: object) -> None:
            raise RuntimeError(msg)

        monkeypatch.setattr(cli, "fetch_benchmark", _fail)
        result = runner.invoke(app, ["fetch", "bird"])
        assert result.exit_code == 1
        assert "hash mismatch!" in result.output


@pytest.mark.unit
class TestDoctor:
    @pytest.fixture(autouse=True)
    def _isolate_platform_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for var in (
            "EVALDATA_DUCKDB_PATH",
            "EVALDATA_POSTGRES_CONNINFO",
            "EVALDATA_SQLITE_PATH",
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
            sqlite=":memory:",
            databricks_server_hostname="h",
            databricks_http_path="/sql/1.0/warehouses/abc",
        )
        assert {ref.kind for ref in refs} == set(get_args(PlatformKind))

    def test_databricks_flags_required_together(self) -> None:
        result = runner.invoke(app, ["doctor", "--databricks-server-hostname", "h"])
        assert result.exit_code == 2
        assert "together" in result.output

    def test_dbt_project_resolves_and_probes(self, tmp_path: Path) -> None:
        project = _copy_dbt_project(tmp_path)
        result = runner.invoke(app, ["doctor", "--dbt-project", str(project)])
        assert result.exit_code == 0
        assert "duckdb" in result.output
        assert "OK" in result.output

    def test_dbt_project_resolution_failure_is_fail(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["doctor", "--dbt-project", str(tmp_path)])
        assert result.exit_code == 1
        assert "FAIL" in result.output


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
