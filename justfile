test *args:
    uv run pytest {{args}}

test-cov *args:
    uv run coverage erase
    uv run coverage run -m pytest {{args}}
    uv run coverage combine
    uv run coverage report
