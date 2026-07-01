"""Unit tests for the `mf`-command wiring: command building, profiles dir, and error paths."""

import subprocess
from pathlib import Path

import pytest

from evaldata.dbt import DbtError, MetricQuery, run
from evaldata.dbt.metricflow import _query_command

pytestmark = pytest.mark.unit


def _fake_mf(monkeypatch: pytest.MonkeyPatch, captured: dict, *, returncode: int = 0) -> None:
    """Stub out `mf` discovery and execution, writing a CSV and recording the call's env/cwd."""
    monkeypatch.setattr("evaldata.dbt.metricflow.shutil.which", lambda _: "mf")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["env"] = kwargs["env"]
        captured["cwd"] = kwargs["cwd"]
        Path(command[command.index("--csv") + 1]).write_text("a,b\n1,2\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, stdout="", stderr="boom")

    monkeypatch.setattr("evaldata.dbt.metricflow.subprocess.run", fake_run)


def test_query_command_includes_all_parts() -> None:
    query = MetricQuery(
        metrics=["revenue", "orders"],
        group_by=["metric_time__month"],
        where=["{{ Dimension('order_id__is_large_order') }} = true"],
        order_by=["-metric_time__month"],
        limit=10,
    )
    command = _query_command("mf", query, Path("/tmp/out.csv"))
    assert command[:4] == ["mf", "query", "--quiet", "--metrics"]
    assert "revenue,orders" in command
    assert command[command.index("--group-by") + 1] == "metric_time__month"
    assert command[command.index("--where") + 1] == "{{ Dimension('order_id__is_large_order') }} = true"
    assert command[command.index("--order") + 1] == "-metric_time__month"
    assert command[command.index("--limit") + 1] == "10"
    assert command[-2:] == ["--csv", "/tmp/out.csv"]


def test_query_command_omits_absent_parts() -> None:
    command = _query_command("mf", MetricQuery(metrics=["revenue"]), Path("/tmp/out.csv"))
    assert "--group-by" not in command
    assert "--where" not in command
    assert "--order" not in command
    assert "--limit" not in command


def test_run_without_mf_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("evaldata.dbt.metricflow.shutil.which", lambda _: None)
    result = run(MetricQuery(metrics=["revenue"]), "target")
    assert isinstance(result, DbtError)
    assert result.kind == "metricflow_unavailable"


def test_run_parses_csv_and_defaults_profiles_to_project(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    _fake_mf(monkeypatch, captured)
    rows = run(MetricQuery(metrics=["revenue"]), "/proj/target")
    assert rows == [{"a": "1", "b": "2"}]
    assert captured["env"]["DBT_PROFILES_DIR"] == str(Path("/proj"))
    assert str(captured["cwd"]) == str(Path("/proj"))


def test_run_honours_explicit_profiles_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    _fake_mf(monkeypatch, captured)
    run(MetricQuery(metrics=["revenue"]), "/proj/target", profiles_dir="/home/me/.dbt")
    assert captured["env"]["DBT_PROFILES_DIR"] == "/home/me/.dbt"


def test_run_reports_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}
    _fake_mf(monkeypatch, captured, returncode=1)
    result = run(MetricQuery(metrics=["revenue"]), "/proj/target")
    assert isinstance(result, DbtError)
    assert result.kind == "metric_query_invalid"
    assert "boom" in result.message
