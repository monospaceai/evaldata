"""Build `EvalCase`s from a dbt project."""

from pathlib import Path
from typing import Annotated, Any, Literal, assert_never

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evaldata.dbt._yaml import read_yaml
from evaldata.dbt.context import DbtContext, DbtTest
from evaldata.dbt.errors import DbtError
from evaldata.types import (
    EvalCase,
    Expectation,
    ExpectationSuite,
    GoldQuery,
    NotNullExpectation,
    PlatformRef,
    UniqueExpectation,
)

Mode = Literal["authored", "model", "tests"]


class _AuthoredCase(BaseModel):
    """One entry in a cases file."""

    model_config = ConfigDict(extra="forbid")

    question: Annotated[str, Field(min_length=1)]
    gold_sql: Annotated[str, Field(min_length=1)]
    select: list[str] | None = None
    id: str | None = None


def load_dbt(
    target_dir: str | Path,
    *,
    platform: PlatformRef,
    cases: str | Path | None = None,
    mode: Mode = "authored",
) -> list[EvalCase] | DbtError:
    """Build eval cases from a built dbt project's artifacts.

    The schema context for each case is the project's tables (sources and models) rendered as
    `CREATE TABLE` statements into `metadata["schema_ddl"]`, ready for a schema-aware solver.

    In `authored` mode (the default), `cases` is a YAML/JSON file of `{question, gold_sql,
    select?, id?}` entries; `select` scopes the schema context to named tables. In `model` mode,
    each documented, compiled model becomes a case whose question is the model's description and
    whose gold query is the model's compiled SQL. In `tests` mode, each documented model with
    `not_null` or `unique` tests becomes a case whose expected outcome is an `ExpectationSuite`
    built from those tests.

    Args:
        target_dir: A dbt `target/` directory holding `manifest.json` (and optionally
            `catalog.json`).
        platform: The warehouse the project is built in; every case runs against it.
        cases: Path to the cases file (required for `authored` mode; ignored for `model`).
        mode: `authored` to read `cases`, `model` to derive cases from documented models, or
            `tests` to build expectation suites from documented models' data tests.

    Returns:
        The eval cases, or a `DbtError` if the artifacts, cases, or mode inputs cannot be read.
    """
    context = DbtContext.from_target_dir(target_dir)
    if isinstance(context, DbtError):
        return context
    match mode:
        case "authored":
            return _authored_cases(context, platform, cases)
        case "model":
            return _model_cases(context, platform)
        case "tests":
            return _test_cases(context, platform)
        case _ as unreachable:  # pragma: no cover - exhaustiveness guard
            assert_never(unreachable)


def _authored_cases(context: DbtContext, platform: PlatformRef, cases: str | Path | None) -> list[EvalCase] | DbtError:
    if cases is None:
        return DbtError(kind="cases_not_found", message="authored mode requires a cases file")
    raw = read_yaml(Path(cases), not_found="cases_not_found", invalid="cases_invalid")
    if isinstance(raw, DbtError):
        return raw
    if not isinstance(raw, list):
        return DbtError(kind="cases_invalid", message=f"{cases} must be a list of cases")

    out: list[EvalCase] = []
    for index, entry in enumerate(raw):
        try:
            spec = _AuthoredCase.model_validate(entry)
        except ValidationError as e:
            return DbtError(kind="cases_invalid", message=f"case {index} is invalid: {e}", cause=e)
        out.append(
            EvalCase(
                id=spec.id or f"dbt/authored/{index}",
                input=spec.question,
                expected=GoldQuery(sql=spec.gold_sql),
                platform=platform,
                metadata=_metadata(context.schema_context(select=spec.select).as_text()),
            )
        )
    return out


def _model_cases(context: DbtContext, platform: PlatformRef) -> list[EvalCase]:
    schema_ddl = context.schema_context().as_text()
    out: list[EvalCase] = []
    for model in context.models():
        if not model.description or not model.compiled_sql:
            continue
        out.append(
            EvalCase(
                id=f"dbt/model/{model.name}",
                input=model.description,
                expected=GoldQuery(sql=model.compiled_sql),
                platform=platform,
                metadata=_metadata(schema_ddl, model=model.name),
            )
        )
    return out


def _expectation_for(test: DbtTest) -> Expectation | None:
    if test.column is None:
        return None
    if test.name == "not_null":
        return NotNullExpectation(column=test.column)
    if test.name == "unique":
        return UniqueExpectation(column=test.column)
    return None


def _test_cases(context: DbtContext, platform: PlatformRef) -> list[EvalCase]:
    suites: dict[str, list[Expectation]] = {}
    for test in context.tests():
        expectation = _expectation_for(test)
        if expectation is not None:
            suites.setdefault(test.model, []).append(expectation)

    schema_ddl = context.schema_context().as_text()
    out: list[EvalCase] = []
    for model in context.models():
        expectations = suites.get(model.name)
        if not model.description or not expectations:
            continue
        out.append(
            EvalCase(
                id=f"dbt/test/{model.name}",
                input=model.description,
                expected=ExpectationSuite(expectations=expectations),
                platform=platform,
                metadata=_metadata(schema_ddl, model=model.name),
            )
        )
    return out


def _metadata(schema_ddl: str, *, model: str | None = None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"source": "dbt", "schema_ddl": schema_ddl}
    if model is not None:
        metadata["model"] = model
    return metadata
