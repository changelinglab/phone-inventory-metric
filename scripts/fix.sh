#!/usr/bin/env bash

set -xe
uv run ruff check --fix src/ tests/
uv run mypy src/ tests/
uv run isort --profile black src/ tests/
uv run black src/ tests/
