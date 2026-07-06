"""Lazy, extra-gated exposure of optional-dependency classes in the public API."""

import builtins
from collections.abc import Callable
from typing import Any

import pytest

import evaldata
from evaldata import llm, platforms, solvers

pytestmark = pytest.mark.unit


def test_prompt_solver_top_level() -> None:
    from evaldata.solvers.prompt import PromptSolver

    assert evaldata.PromptSolver is PromptSolver
    assert "PromptSolver" in dir(evaldata)


def test_prompt_solver_subpackage() -> None:
    from evaldata.solvers.prompt import PromptSolver

    assert solvers.PromptSolver is PromptSolver
    assert "PromptSolver" in dir(solvers)


def test_postgres_adapter_subpackage() -> None:
    from evaldata.platforms.postgres import PostgresAdapter

    assert platforms.PostgresAdapter is PostgresAdapter
    assert "PostgresAdapter" in dir(platforms)


def test_postgres_adapter_not_top_level() -> None:
    with pytest.raises(AttributeError):
        _ = evaldata.PostgresAdapter


def test_databricks_adapter_subpackage() -> None:
    from evaldata.platforms.databricks import DatabricksAdapter

    assert platforms.DatabricksAdapter is DatabricksAdapter
    assert "DatabricksAdapter" in dir(platforms)


def test_databricks_adapter_not_top_level() -> None:
    with pytest.raises(AttributeError):
        _ = evaldata.DatabricksAdapter


def test_snowflake_adapter_subpackage() -> None:
    from evaldata.platforms.snowflake import SnowflakeAdapter

    assert platforms.SnowflakeAdapter is SnowflakeAdapter
    assert "SnowflakeAdapter" in dir(platforms)


def test_snowflake_adapter_not_top_level() -> None:
    with pytest.raises(AttributeError):
        _ = evaldata.SnowflakeAdapter


def _blocking_import(blocked: str) -> Callable[..., Any]:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked:
            msg = f"No module named {blocked!r}"
            raise ModuleNotFoundError(msg)
        return real_import(name, *args, **kwargs)

    return fake_import


def test_lite_llm_subpackage() -> None:
    from evaldata.llm.litellm import LiteLlm

    assert llm.LiteLlm is LiteLlm
    assert "LiteLlm" in dir(llm)
    assert evaldata.LiteLlm is LiteLlm
    assert "LiteLlm" in dir(evaldata)


def test_lite_llm_missing_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "evaldata.llm.litellm", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import("litellm"))
    with pytest.raises(ImportError, match=r"evaldata\[litellm\]"):
        llm.__getattr__("LiteLlm")


def test_postgres_adapter_missing_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "evaldata.platforms.postgres", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import("psycopg"))
    with pytest.raises(ImportError, match=r"evaldata\[postgres\]"):
        platforms.__getattr__("PostgresAdapter")


def test_databricks_adapter_missing_databricks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "evaldata.platforms.databricks", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import("databricks.sql"))
    with pytest.raises(ImportError, match=r"evaldata\[databricks\]"):
        platforms.__getattr__("DatabricksAdapter")


def test_snowflake_adapter_missing_snowflake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "evaldata.platforms.snowflake", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import("snowflake.connector"))
    with pytest.raises(ImportError, match=r"evaldata\[snowflake\]"):
        platforms.__getattr__("SnowflakeAdapter")


def test_unknown_attribute_top_level() -> None:
    with pytest.raises(AttributeError):
        _ = evaldata.NoSuchThing


def test_unknown_attribute_solvers() -> None:
    with pytest.raises(AttributeError):
        _ = solvers.NoSuchThing


def test_unknown_attribute_platforms() -> None:
    with pytest.raises(AttributeError):
        _ = platforms.NoSuchThing
