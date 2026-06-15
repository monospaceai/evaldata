"""Lazy, extra-gated exposure of optional-dependency classes in the public API."""

import builtins
from collections.abc import Callable
from typing import Any

import pytest

import dataeval
from dataeval import platforms, solvers


def test_prompt_solver_top_level() -> None:
    from dataeval.solvers.prompt import PromptSolver

    assert dataeval.PromptSolver is PromptSolver
    assert "PromptSolver" in dir(dataeval)


def test_prompt_solver_subpackage() -> None:
    from dataeval.solvers.prompt import PromptSolver

    assert solvers.PromptSolver is PromptSolver
    assert "PromptSolver" in dir(solvers)


def test_postgres_adapter_subpackage() -> None:
    from dataeval.platforms.postgres import PostgresAdapter

    assert platforms.PostgresAdapter is PostgresAdapter
    assert "PostgresAdapter" in dir(platforms)


def test_postgres_adapter_not_top_level() -> None:
    with pytest.raises(AttributeError):
        _ = dataeval.PostgresAdapter


def _blocking_import(blocked: str) -> Callable[..., Any]:
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == blocked:
            msg = f"No module named {blocked!r}"
            raise ModuleNotFoundError(msg)
        return real_import(name, *args, **kwargs)

    return fake_import


def test_prompt_solver_missing_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "dataeval.solvers.prompt", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import("litellm"))
    with pytest.raises(ImportError, match=r"dataeval\[litellm\]"):
        solvers.__getattr__("PromptSolver")


def test_postgres_adapter_missing_psycopg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(__import__("sys").modules, "dataeval.platforms.postgres", raising=False)
    monkeypatch.setattr(builtins, "__import__", _blocking_import("psycopg"))
    with pytest.raises(ImportError, match=r"dataeval\[postgres\]"):
        platforms.__getattr__("PostgresAdapter")


def test_unknown_attribute_top_level() -> None:
    with pytest.raises(AttributeError):
        _ = dataeval.NoSuchThing


def test_unknown_attribute_solvers() -> None:
    with pytest.raises(AttributeError):
        _ = solvers.NoSuchThing


def test_unknown_attribute_platforms() -> None:
    with pytest.raises(AttributeError):
        _ = platforms.NoSuchThing
