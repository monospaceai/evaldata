"""Read and validate a dbt project's artifacts (manifest, catalog, semantic manifest).

The manifest schema version is validated before any fields are read.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaldata.dbt.errors import DbtError

# Fields used here are present and stable from schema v10 onward.
MIN_MANIFEST_VERSION = 10


@dataclass(frozen=True)
class Artifacts:
    """A dbt project's parsed artifacts: the manifest, optional catalog, and schema version.

    `catalog` is `None` when the project has no `catalog.json` (no `dbt docs generate` was run).
    `semantic_manifest` is `None` when the project has no `semantic_manifest.json` (no semantic
    layer, or `dbt parse` was not run).
    """

    manifest: dict[str, Any]
    catalog: dict[str, Any] | None
    semantic_manifest: dict[str, Any] | None
    schema_version: str


def _read_json_object(path: Path) -> dict[str, Any] | DbtError:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return DbtError(kind="artifact_invalid", message=f"could not read {path}: {e}", cause=e)
    if not isinstance(data, dict):
        return DbtError(kind="artifact_invalid", message=f"{path} is not a JSON object")
    return data


def _schema_version(manifest: dict[str, Any]) -> str | None:
    metadata = manifest.get("metadata")
    if not isinstance(metadata, dict):
        return None
    url = metadata.get("dbt_schema_version")
    if not isinstance(url, str) or not url:
        return None
    return url.rsplit("/", 1)[-1].removesuffix(".json")


def _version_number(token: str) -> int | None:
    digits = token[1:] if token.startswith("v") else token
    return int(digits) if digits.isdigit() else None


def read_artifacts(target_dir: str | Path) -> Artifacts | DbtError:
    """Read and validate the dbt artifacts in a `target/` directory.

    Reads `manifest.json` (required), `catalog.json`, and `semantic_manifest.json` (both
    optional), and validates the manifest schema version.

    Args:
        target_dir: Path to a dbt `target/` directory.

    Returns:
        An `Artifacts` on success, or a `DbtError` if the manifest is absent
        (`target_not_found`), an artifact is unreadable or malformed (`artifact_invalid`), or the
        manifest schema version is unsupported (`unsupported_schema_version`).
    """
    target = Path(target_dir)
    manifest_path = target / "manifest.json"
    if not manifest_path.is_file():
        return DbtError(kind="target_not_found", message=f"no manifest.json in {target}")

    manifest = _read_json_object(manifest_path)
    if isinstance(manifest, DbtError):
        return manifest

    version = _schema_version(manifest)
    if version is None:
        return DbtError(kind="artifact_invalid", message=f"{manifest_path} has no metadata.dbt_schema_version")
    number = _version_number(version)
    if number is None or number < MIN_MANIFEST_VERSION:
        return DbtError(
            kind="unsupported_schema_version",
            message=f"unsupported dbt manifest schema version {version!r}; need v{MIN_MANIFEST_VERSION} or newer",
        )

    catalog: dict[str, Any] | None = None
    catalog_path = target / "catalog.json"
    if catalog_path.is_file():
        read = _read_json_object(catalog_path)
        if isinstance(read, DbtError):
            return read
        catalog = read

    semantic_manifest: dict[str, Any] | None = None
    semantic_path = target / "semantic_manifest.json"
    if semantic_path.is_file():
        read = _read_json_object(semantic_path)
        if isinstance(read, DbtError):
            return read
        semantic_manifest = read

    return Artifacts(manifest=manifest, catalog=catalog, semantic_manifest=semantic_manifest, schema_version=version)
