#!/usr/bin/env bash
# Rebuild the ACME fixture and refresh the committed semantic manifest.
#
# The warehouse (acme.duckdb) and target/ are gitignored and rebuilt from the seeds by the e2e;
# only artifacts/semantic_manifest.json is committed, for the hermetic corpus test. Run this when
# the project's models or semantic layer change.
# Requires the `fixtures` dependency group (dbt-duckdb): `uv run --group fixtures bash regen.sh`.
set -euo pipefail
cd "$(dirname "$0")"
export DBT_PROFILES_DIR="$PWD"

rm -f acme.duckdb
rm -rf target logs dbt_packages

dbt seed --profiles-dir "$PWD"
dbt build --profiles-dir "$PWD"
dbt parse --profiles-dir "$PWD"        # writes target/semantic_manifest.json

mkdir -p artifacts
cp target/semantic_manifest.json artifacts/semantic_manifest.json

# Normalise the volatile metadata nested under project_configuration so the artifact is deterministic.
python - <<'PY'
import json
import pathlib

placeholders = {
    "invocation_id": "00000000-0000-0000-0000-000000000000",
    "invocation_started_at": "1970-01-01T00:00:00.000000+00:00",
    "generated_at": "1970-01-01T00:00:00.000000Z",
    "run_started_at": "1970-01-01T00:00:00.000000+00:00",
}
path = pathlib.Path("artifacts/semantic_manifest.json")
doc = json.loads(path.read_text())
metadata = (doc.get("project_configuration") or {}).get("metadata")
if isinstance(metadata, dict):
    for key, value in placeholders.items():
        if key in metadata:
            metadata[key] = value
path.write_text(json.dumps(doc))
PY

echo "regenerated: artifacts/semantic_manifest.json"
