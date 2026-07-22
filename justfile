# Run tests with all optional dependencies and dbt fixtures.
test *args:
    uv run --all-extras --group fixtures pytest {{args}}

test-cov *args:
    uv run coverage erase
    uv run --all-extras --group fixtures coverage run -m pytest {{args}}
    uv run coverage combine
    uv run coverage report

test-cloud *args:
    uv run --all-extras --group fixtures pytest -m cloud {{args}}

# Run non-cloud tests with coverage.
test-gate *args:
    just test-cov '-m "not cloud and not manual"' {{args}}

test-databricks *args:
    just test '-m "databricks and not manual"' {{args}}

test-snowflake *args:
    just test '-m "snowflake and not manual"' {{args}}

test-bigquery *args:
    just test '-m "bigquery and not manual"' {{args}}

# Run manual tests other than Cortex Analyst.
test-manual *args:
    just test '-m "manual and not cortex"' {{args}}

test-cortex *args:
    uv run --all-extras --group fixtures pytest -m cortex {{args}}

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

# Lint, type-check, and run non-cloud tests with coverage.
check: lint typecheck
    just test-gate

# Fast iteration: like `check` but without coverage.
check-nocloud: lint typecheck
    just test '-m "not cloud and not manual"'

_ci: docs-build check
    just test-databricks
    just test-snowflake
    just test-bigquery
    just build

ci:
    UV_FROZEN=1 just _ci

release *args="auto":
    changie batch {{args}}
    changie merge
