#!/usr/bin/env bash

set -xe
.venv/bin/ruff check --fix src/
.venv/bin/mypy -p src.phone_inventory_metric
.venv/bin/isort --profile black src/
.venv/bin/black src/
