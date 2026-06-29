"""Types and `DbtContext` for working with a dbt project's models, sources, and schema."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaldata.dbt.artifacts import read_artifacts
from evaldata.dbt.errors import DbtError


@dataclass(frozen=True)
class Relation:
    """A warehouse relation: its database/schema/identifier parts and dbt's quoted name."""

    database: str
    schema: str
    identifier: str
    quoted: str

    def __str__(self) -> str:
        """Return the warehouse-quoted, fully-qualified relation name."""
        return self.quoted


@dataclass(frozen=True)
class Column:
    """A table column: its name, resolved SQL type (when known), and description (when documented)."""

    name: str
    type: str | None
    description: str | None


@dataclass(frozen=True)
class TableSchema:
    """A queryable table — a dbt model or source — as name, relation, columns, and description."""

    name: str
    relation: Relation
    columns: tuple[Column, ...]
    description: str | None


@dataclass(frozen=True)
class ModelRef:
    """A dbt model: its name, unique id, target relation, compiled SQL, columns, and description."""

    name: str
    unique_id: str
    relation: Relation
    compiled_sql: str | None
    description: str | None
    columns: tuple[Column, ...]


@dataclass(frozen=True)
class SourceRef:
    """A dbt source table: its table name, source collection, relation, columns, and description."""

    name: str
    source_name: str
    relation: Relation
    description: str | None
    columns: tuple[Column, ...]


@dataclass(frozen=True)
class DbtTest:
    """A dbt data test: its type, the model it guards, and the column it targets (if any)."""

    name: str
    model: str
    column: str | None


@dataclass(frozen=True)
class SchemaContext:
    """A selection of tables rendered as schema context for a text-to-SQL prompt."""

    tables: tuple[TableSchema, ...]

    def as_text(self) -> str:
        """Render the tables as `CREATE TABLE` statements for prompt injection.

        Each table renders its description as a leading comment (when documented), its quoted
        relation name, and one line per column (`name type`, with the column description as a
        trailing comment when documented). Tables are separated by a blank line.

        Returns:
            The rendered schema text, or the empty string when there are no tables.
        """
        blocks: list[str] = []
        for table in self.tables:
            lines: list[str] = []
            if table.description:
                lines.append(f"-- {table.description}")
            lines.append(f"CREATE TABLE {table.relation.quoted} (")
            last = len(table.columns) - 1
            for index, column in enumerate(table.columns):
                typed = f" {column.type}" if column.type else ""
                comma = "," if index < last else ""
                comment = f"  -- {column.description}" if column.description else ""
                lines.append(f"  {column.name}{typed}{comma}{comment}")
            lines.append(");")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)


def _relation(node: dict[str, Any]) -> Relation:
    identifier = node["alias"] if "alias" in node else node["identifier"]
    return Relation(
        database=node["database"],
        schema=node["schema"],
        identifier=identifier,
        quoted=node["relation_name"],
    )


def _columns(manifest_columns: dict[str, Any], catalog_columns: dict[str, Any] | None) -> tuple[Column, ...]:
    if catalog_columns:
        ordered = sorted(catalog_columns.items(), key=lambda item: item[1]["index"])
        return tuple(
            Column(
                name=name,
                type=info["type"],
                description=manifest_columns.get(name, {}).get("description") or None,
            )
            for name, info in ordered
        )
    return tuple(
        Column(name=name, type=info.get("data_type"), description=info.get("description") or None)
        for name, info in manifest_columns.items()
    )


def _model_table(model: ModelRef) -> TableSchema:
    return TableSchema(name=model.name, relation=model.relation, columns=model.columns, description=model.description)


def _source_table(source: SourceRef) -> TableSchema:
    return TableSchema(
        name=source.name, relation=source.relation, columns=source.columns, description=source.description
    )


class DbtContext:
    """A dbt project's models, sources, and schema, normalised from its `target/` artifacts.

    Build one with `from_target_dir`. Models are addressable by short name or unique id; column
    types come from `catalog.json` (the resolved warehouse types) when present, falling back to
    the manifest's declared types otherwise.
    """

    def __init__(
        self,
        *,
        models: Iterable[ModelRef],
        sources: Iterable[SourceRef],
        tests: Iterable[DbtTest],
        schema_version: str,
    ) -> None:
        """Build a context from pre-built models, sources, and tests.

        Args:
            models: The project's models.
            sources: The project's source tables.
            tests: The project's data tests.
            schema_version: The manifest schema version the parts were read from.
        """
        self._models = tuple(models)
        self._sources = tuple(sources)
        self._tests = tuple(tests)
        self._schema_version = schema_version
        self._by_key: dict[str, ModelRef] = {}
        for model in self._models:
            self._by_key[model.unique_id] = model
            self._by_key[model.name] = model

    @classmethod
    def from_target_dir(cls, path: str | Path) -> "DbtContext | DbtError":
        """Build a context from a dbt `target/` directory.

        Args:
            path: Path to a dbt `target/` directory holding `manifest.json` (and optionally
                `catalog.json`).

        Returns:
            A `DbtContext` on success, or a `DbtError` if the artifacts are missing, malformed,
            or an unsupported schema version.
        """
        artifacts = read_artifacts(path)
        if isinstance(artifacts, DbtError):
            return artifacts

        manifest, catalog = artifacts.manifest, artifacts.catalog
        catalog_nodes = (catalog or {}).get("nodes", {})
        catalog_sources = (catalog or {}).get("sources", {})

        models: list[ModelRef] = []
        for uid, node in manifest["nodes"].items():
            if node.get("resource_type") != "model":
                continue
            cataloged = catalog_nodes.get(uid)
            models.append(
                ModelRef(
                    name=node["name"],
                    unique_id=uid,
                    relation=_relation(node),
                    compiled_sql=node.get("compiled_code"),
                    description=node.get("description") or None,
                    columns=_columns(node.get("columns", {}), cataloged.get("columns") if cataloged else None),
                )
            )

        sources: list[SourceRef] = []
        for uid, node in manifest["sources"].items():
            cataloged = catalog_sources.get(uid)
            sources.append(
                SourceRef(
                    name=node["name"],
                    source_name=node["source_name"],
                    relation=_relation(node),
                    description=node.get("description") or None,
                    columns=_columns(node.get("columns", {}), cataloged.get("columns") if cataloged else None),
                )
            )

        name_by_uid = {model.unique_id: model.name for model in models}
        tests: list[DbtTest] = []
        for node in manifest["nodes"].values():
            if node.get("resource_type") != "test":
                continue
            metadata = node.get("test_metadata")
            if not isinstance(metadata, dict):
                continue
            model_name = name_by_uid.get(node.get("attached_node"))
            if model_name is None:
                continue
            tests.append(DbtTest(name=metadata["name"], model=model_name, column=node.get("column_name")))

        return cls(models=models, sources=sources, tests=tests, schema_version=artifacts.schema_version)

    def model(self, name_or_uid: str) -> ModelRef | None:
        """Return the model addressed by short name or unique id, or `None` if there is none.

        Args:
            name_or_uid: A model's short name (e.g. `customers`) or unique id (e.g.
                `model.my_project.customers`).

        Returns:
            The matching `ModelRef`, or `None`.
        """
        return self._by_key.get(name_or_uid)

    def compiled_sql(self, name_or_uid: str) -> str | None:
        """Return a model's compiled SQL, or `None` if the model or its compiled SQL is absent.

        Args:
            name_or_uid: A model's short name or unique id.

        Returns:
            The model's `compiled_code`, or `None`.
        """
        model = self.model(name_or_uid)
        return model.compiled_sql if model is not None else None

    def relation(self, name_or_uid: str) -> Relation | None:
        """Return a model's target relation, or `None` if there is no such model.

        Args:
            name_or_uid: A model's short name or unique id.

        Returns:
            The model's `Relation`, or `None`.
        """
        model = self.model(name_or_uid)
        return model.relation if model is not None else None

    def tables(self) -> list[TableSchema]:
        """Return every queryable table — sources then models — as table schemas.

        Returns:
            The source tables followed by the model tables.
        """
        return [_source_table(s) for s in self._sources] + [_model_table(m) for m in self._models]

    def models(self) -> list[ModelRef]:
        """Return the project's models.

        Returns:
            The models, in manifest order.
        """
        return list(self._models)

    def tests(self) -> list[DbtTest]:
        """Return the project's data tests attached to models.

        Returns:
            The data tests, in manifest order.
        """
        return list(self._tests)

    def sources(self) -> list[SourceRef]:
        """Return the project's source tables.

        Returns:
            The source tables, in manifest order.
        """
        return list(self._sources)

    def schema_version(self) -> str:
        """Return the manifest schema version the project was read from.

        Returns:
            The schema version token (e.g. `v12`).
        """
        return self._schema_version

    def schema_context(
        self,
        *,
        include_sources: bool = True,
        include_models: bool = True,
        select: Iterable[str] | None = None,
    ) -> SchemaContext:
        """Build schema context for a text-to-SQL prompt from the project's tables.

        Args:
            include_sources: Include the project's source tables.
            include_models: Include the project's models.
            select: If given, keep only tables whose name is in this collection.

        Returns:
            A `SchemaContext` over the selected tables.
        """
        tables: list[TableSchema] = []
        if include_sources:
            tables.extend(_source_table(s) for s in self._sources)
        if include_models:
            tables.extend(_model_table(m) for m in self._models)
        if select is not None:
            names = set(select)
            tables = [t for t in tables if t.name in names]
        return SchemaContext(tables=tuple(tables))
