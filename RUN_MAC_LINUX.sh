#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PYTHON=""
if command -v python3 >/dev/null 2>&1; then PYTHON=python3; elif command -v python >/dev/null 2>&1; then PYTHON=python; else echo "Python 3.11+ is required."; exit 1; fi
if [ ! -x .venv/bin/python ]; then "$PYTHON" -m venv .venv; fi
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
