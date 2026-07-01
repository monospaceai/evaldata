"""Unit tests for the `mf`-command wiring: command building and the missing-toolchain path."""

from pathlib import Path

import pytest

from evaldata.dbt import DbtError, MetricQuery, run
from evaldata.dbt.metricflow import _query_command

pytestmark = pytest.mark.unit


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
