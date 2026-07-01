# The `fixtures` group carries the dbt toolchain (dbt-metricflow[dbt-duckdb]) the Semantic
# Layer e2e runs against; the other extras cover the rest of the suite.
test *args:
    uv run --all-extras --group fixtures pytest {{args}}

test-cov *args:
    uv run coverage erase
    uv run --all-extras --group fixtures coverage run -m pytest {{args}}
    uv run coverage combine
    uv run coverage report

# Run only `cloud` e2e (Databricks, …) in isolation; needs the secrets in the env.
test-cloud *args:
    uv run --all-extras --group fixtures pytest -m cloud {{args}}

# Regenerate the checked-in dbt fixture artifacts (needs the `fixtures` group: dbt-duckdb).
dbt-fixture:
    uv run --group fixtures bash tests/dbt/fixtures/jaffle_duckdb/regen.sh

lint:
    uv run ruff check
    uv run ruff format --check

fix:
    uv run ruff check --fix
    uv run ruff format

typecheck:
    uv run --all-extras ty check src examples

precommit:
    uv run pre-commit run --all-files --show-diff-on-failure

build:
    uv build

# Live-reload docs at http://127.0.0.1:8000
docs-serve:
    uv run --group docs mkdocs serve

# Build the docs into ./site, failing on broken references.
docs-build:
    uv run --group docs mkdocs build --strict

# Publish versioned docs to the gh-pages branch. Pass the version, e.g. `just docs-deploy 0.1`.
docs-deploy version:
    uv run --group docs mike deploy --push --update-aliases {{version}} latest
    uv run --group docs mike set-default --push latest

# Everyday gate: runs everything incl. `cloud` (needs credentials in the env); coverage 100%.
check: lint typecheck
    just test-cov

# Fast iteration: like `check` but skips `cloud`; no coverage gate. CI still runs everything.
check-nocloud: lint typecheck
    just test '-m "not cloud"'

ci: check build

release *args="auto":
    changie batch {{args}}
    changie merge
