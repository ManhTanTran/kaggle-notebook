#!/usr/bin/env bash
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m ipykernel install --user --name chunkbench --display-name "Python (chunkbench)"
python -m pytest
echo "Select the Python (chunkbench) kernel in your editor."

