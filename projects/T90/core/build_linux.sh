#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${SCRIPT_DIR}"

echo "[1/3] Installing build dependencies"
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install "setuptools>=68" "wheel>=0.41" "Cython>=3,<4"

echo "[2/3] Building Cython extensions in place"
"${PYTHON_BIN}" setup.py build_ext --inplace

echo "[3/3] Cleaning build cache"
rm -rf build
find "${SCRIPT_DIR}" -maxdepth 1 \( -name "*.c" -o -name "*.html" \) -delete

echo "T90 core build completed."
