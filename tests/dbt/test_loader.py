"""Tests for building EvalCases from a dbt project."""

import textwrap
from pathlib import Path

import pytest

from evaldata.dbt.context import DbtContext, DbtTest, ModelRef, Relation
from evaldata.dbt.errors import DbtError
from evaldata.dbt.loader import _expectation_for, _model_cases, _test_cases, load_dbt, load_dbt_metrics
from evaldata.platforms.registry import duckdb_platform
from evaldata.types import ExpectationSuite, GoldQuery

pytestmark = pytest.mark.unit

ARTIFACTS = Path(__file__).parent / "fixtures" / "jaffle_duckdb" / "artifacts"
PLATFORM = duckdb_platform(name="dbt-test", path=":memory:")


def _write_cases(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "cases.yml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_authored_cases(tmp_path: Path) -> None:
    cases_file = _write_cases(
        tmp_path,
        """
        - question: How many customers?
          gold_sql: select count(*) from customers
          select: [customers]
          id: count-customers
        - question: List orders
          gold_sql: select * from stg_orders
        """,
    )
    cases = load_dbt(ARTIFACTS, platform=PLATFORM, cases=cases_file)
    assert not isinstance(cases, DbtError)
    assert [c.id for c in cases] == ["count-customers", "dbt/authored/1"]

    first = cases[0]
    assert first.input == "How many customers?"
    assert isinstance(first.expected, GoldQuery)
    assert first.expected.sql == "select count(*) from customers"
    assert first.platform is PLATFORM
    assert first.metadata["source"] == "dbt"
    assert "model" not in first.metadata
    assert 'CREATE TABLE "jaffle"."main"."customers"' in first.metadata["schema_ddl"]
    assert "raw_customers" not in first.metadata["schema_ddl"]
    assert "raw_customers" in cases[1].metadata["schema_ddl"]


def test_model_mode(tmp_path: Path) -> None:
    cases = load_dbt(ARTIFACTS, platform=PLATFORM, mode="model")
    assert not isinstance(cases, DbtError)
    assert [c.id for c in cases] == [
        "dbt/model/stg_customers",
        "dbt/model/stg_orders",
        "dbt/model/all_days",
        "dbt/model/customers",
    ]
    customers = cases[-1]
    assert customers.input == "Customer dimension enriched with order activity."
    assert isinstance(customers.expected, GoldQuery)
    assert "select" in customers.expected.sql.lower()
    assert customers.metadata["model"] == "customers"
    assert customers.metadata["source"] == "dbt"


def test_load_dbt_metrics(tmp_path: Path) -> None:
    cases_file = _write_cases(
        tmp_path,
        """
        - question: Revenue by month?
          metrics: [revenue]
          group_by: [metric_time__month]
          order_by: ["-metric_time__month"]
          limit: 12
          id: revenue-by-month
        - question: How many orders?
          metrics: [order_count]
        """,
    )
    cases = load_dbt_metrics(ARTIFACTS, platform=PLATFORM, cases=cases_file, profiles_dir="/home/me/.dbt")
    assert not isinstance(cases, DbtError)
    assert [c.id for c in cases] == ["revenue-by-month", "dbt/metrics/1"]

    first = cases[0]
    assert first.input == "Revenue by month?"
    assert first.gold.metrics == ["revenue"]
    assert first.gold.group_by == ["metric_time__month"]
    assert first.gold.order_by == ["-metric_time__month"]
    assert first.gold.limit == 12
    assert first.target_dir == str(ARTIFACTS)
    assert first.profiles_dir == "/home/me/.dbt"
    assert "revenue" in first.sl_context


def test_load_dbt_metrics_bad_target_dir(tmp_path: Path) -> None:
    result = load_dbt_metrics(tmp_path, platform=PLATFORM, cases=tmp_path / "cases.yml")
    assert isinstance(result, DbtError)
    assert result.kind == "target_not_found"


def test_load_dbt_metrics_cases_file_missing(tmp_path: Path) -> None:
    result = load_dbt_metrics(ARTIFACTS, platform=PLATFORM, cases=tmp_path / "nope.yml")
    assert isinstance(result, DbtError)
    assert result.kind == "cases_not_found"


def test_load_dbt_metrics_cases_not_a_list(tmp_path: Path) -> None:
    result = load_dbt_metrics(ARTIFACTS, platform=PLATFORM, cases=_write_cases(tmp_path, "metrics: [x]\n"))
    assert isinstance(result, DbtError)
    assert result.kind == "cases_invalid"


def test_load_dbt_metrics_invalid_entry(tmp_path: Path) -> None:
    result = load_dbt_metrics(ARTIFACTS, platform=PLATFORM, cases=_write_cases(tmp_path, "- question: no metrics\n"))
    assert isinstance(result, DbtError)
    assert result.kind == "cases_invalid"


def test_bad_target_dir(tmp_path: Path) -> None:
    result = load_dbt(tmp_path, platform=PLATFORM, mode="model")
    assert isinstance(result, DbtError)
    assert result.kind == "target_not_found"


def test_authored_requires_cases() -> None:
    result = load_dbt(ARTIFACTS, platform=PLATFORM)
    assert isinstance(result, DbtError)
    assert result.kind == "cases_not_found"


def test_cases_file_missing(tmp_path: Path) -> None:
    result = load_dbt(ARTIFACTS, platform=PLATFORM, cases=tmp_path / "nope.yml")
    assert isinstance(result, DbtError)
    assert result.kind == "cases_not_found"


def test_cases_not_a_list(tmp_path: Path) -> None:
    result = load_dbt(ARTIFACTS, platform=PLATFORM, cases=_write_cases(tmp_path, "question: x\n"))
    assert isinstance(result, DbtError)
    assert result.kind == "cases_invalid"


def test_invalid_case_entry(tmp_path: Path) -> None:
    result = load_dbt(ARTIFACTS, platform=PLATFORM, cases=_write_cases(tmp_path, "- question: no sql here\n"))
    assert isinstance(result, DbtError)
    assert result.kind == "cases_invalid"


def test_malformed_cases_yaml(tmp_path: Path) -> None:
    result = load_dbt(ARTIFACTS, platform=PLATFORM, cases=_write_cases(tmp_path, "- question: [unclosed\n"))
    assert isinstance(result, DbtError)
    assert result.kind == "cases_invalid"


def test_model_mode_skips_undocumented_or_uncompiled() -> None:
    relation = Relation("db", "sc", "m", '"db"."sc"."m"')
    documented = ModelRef("m1", "model.x.m1", relation, "select 1", "documented", ())
    no_description = ModelRef("m2", "model.x.m2", relation, "select 2", None, ())
    no_compiled_sql = ModelRef("m3", "model.x.m3", relation, None, "documented but not compiled", ())
    context = DbtContext(
        models=[documented, no_description, no_compiled_sql], sources=[], tests=[], schema_version="v12"
    )

    cases = _model_cases(context, PLATFORM)
    assert [c.id for c in cases] == ["dbt/model/m1"]


def test_tests_mode(tmp_path: Path) -> None:
    cases = load_dbt(ARTIFACTS, platform=PLATFORM, mode="tests")
    assert not isinstance(cases, DbtError)
    assert [c.id for c in cases] == ["dbt/test/customers"]
    suite = cases[0].expected
    assert isinstance(suite, ExpectationSuite)
    assert sorted(e.kind for e in suite.expectations) == ["not_null", "unique"]
    assert cases[0].metadata["model"] == "customers"


def test_expectation_for_maps_supported_tests() -> None:
    assert _expectation_for(DbtTest("not_null", "m", "c")).kind == "not_null"
    assert _expectation_for(DbtTest("unique", "m", "c")).kind == "unique"
    assert _expectation_for(DbtTest("relationships", "m", "c")) is None
    assert _expectation_for(DbtTest("not_null", "m", None)) is None


def test_test_cases_filters_models_and_unsupported_tests() -> None:
    relation = Relation("db", "sc", "m", '"db"."sc"."m"')
    with_tests = ModelRef("m1", "model.x.m1", relation, "select 1", "documented", ())
    no_tests = ModelRef("m2", "model.x.m2", relation, "select 2", "documented", ())
    undocumented = ModelRef("m3", "model.x.m3", relation, "select 3", None, ())
    tests = [DbtTest("not_null", "m1", "c"), DbtTest("relationships", "m1", "c"), DbtTest("not_null", "m3", "c")]
    context = DbtContext(models=[with_tests, no_tests, undocumented], sources=[], tests=tests, schema_version="v12")

    cases = _test_cases(context, PLATFORM)
    assert [c.id for c in cases] == ["dbt/test/m1"]
    suite = cases[0].expected
    assert isinstance(suite, ExpectationSuite)
    assert [e.kind for e in suite.expectations] == ["not_null"]
