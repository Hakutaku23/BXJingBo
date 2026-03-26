#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
CORE_DIR="${PROJECT_DIR}/core"
CONFIG_DIR="${PROJECT_DIR}/config"
ASSETS_DIR="${PROJECT_DIR}/assets"

OUTPUT_DIR="${PROJECT_DIR}/dist"
PACKAGE_NAME="T90_delivery"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SKIP_BUILD=0
KEEP_STAGE=0
required_modules=("window_encoder" "casebase" "runtime_config" "online_recommender" "stage_aware_recommender")

usage() {
  cat <<'EOF'
Usage:
  ./package_delivery.sh [--output-dir DIR] [--package-name NAME] [--python PYTHON_BIN] [--skip-build] [--keep-stage]

Options:
  --output-dir DIR     Output directory for the staged package and tar.gz archive.
  --package-name NAME  Base name for the staged package and archive.
  --python BIN         Python executable used for building the core extensions.
  --skip-build         Skip the Cython build step and use existing compiled core artifacts.
  --keep-stage         Keep the staged delivery directory after creating the tar.gz archive.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --package-name)
      PACKAGE_NAME="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=1
      shift
      ;;
    --keep-stage)
      KEEP_STAGE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

if [[ "${SKIP_BUILD}" -eq 0 ]]; then
  echo "[1/6] Building core extensions"
  if [[ ! -f "${CORE_DIR}/build_linux.sh" ]]; then
    echo "Missing core/build_linux.sh in ${CORE_DIR}." >&2
    exit 1
  fi
  (
    cd "${CORE_DIR}"
    PYTHON_BIN="${PYTHON_BIN}" bash ./build_linux.sh
  )
else
  echo "[1/6] Skipping core build"
fi

echo "[2/6] Checking compiled core artifacts"
compiled_core=()
for module_name in "${required_modules[@]}"; do
  shopt -s nullglob
  module_matches=("${CORE_DIR}/${module_name}"*.so "${CORE_DIR}/${module_name}"*.pyd)
  shopt -u nullglob
  if [[ ${#module_matches[@]} -eq 0 ]]; then
    echo "Missing compiled artifact for core module: ${module_name}" >&2
    echo "Check the Linux build environment or rerun without --skip-build." >&2
    exit 1
  fi
  compiled_core+=("${module_matches[@]}")
done

timestamp="$(date +%Y%m%d_%H%M%S)"
stage_dir="${OUTPUT_DIR}/${PACKAGE_NAME}_${timestamp}"
package_root="${stage_dir}/T90"
archive_path="${OUTPUT_DIR}/${PACKAGE_NAME}_${timestamp}.tar.gz"

echo "[3/6] Preparing staged delivery directory"
rm -rf "${stage_dir}"
mkdir -p "${package_root}/core" "${package_root}/config" "${package_root}/assets"

cp "${PROJECT_DIR}/interface.py" "${package_root}/"
cp "${PROJECT_DIR}/example.py" "${package_root}/"
cp "${PROJECT_DIR}/README.md" "${package_root}/"
cp "${CONFIG_DIR}/t90_runtime.yaml" "${package_root}/config/"
cp "${ASSETS_DIR}/t90_casebase.csv" "${package_root}/assets/"
cp "${ASSETS_DIR}/t90_casebase_ph120.csv" "${package_root}/assets/"
cp "${ASSETS_DIR}/t90_stage_policy.json" "${package_root}/assets/"
if [[ -f "${ASSETS_DIR}/t90_casebase.parquet" ]]; then
  cp "${ASSETS_DIR}/t90_casebase.parquet" "${package_root}/assets/"
fi
if [[ -f "${ASSETS_DIR}/t90_casebase_ph120.parquet" ]]; then
  cp "${ASSETS_DIR}/t90_casebase_ph120.parquet" "${package_root}/assets/"
fi
if [[ -f "${ASSETS_DIR}/README.md" ]]; then
  cp "${ASSETS_DIR}/README.md" "${package_root}/assets/"
fi

cp "${CORE_DIR}/__init__.py" "${package_root}/core/"
for source_file in "${required_modules[@]}"; do
  cp "${CORE_DIR}/${source_file}.py" "${package_root}/core/"
done
cp "${CORE_DIR}/setup.py" "${package_root}/core/"
cp "${CORE_DIR}/build_linux.sh" "${package_root}/core/"
for artifact in "${compiled_core[@]}"; do
  cp "${artifact}" "${package_root}/core/"
done

cat > "${package_root}/DELIVERY_CONTENTS.txt" <<EOF
T90 delivery package generated at ${timestamp}

Included runtime files:
- interface.py
- example.py
- README.md
- config/t90_runtime.yaml
- assets/t90_casebase.csv
- assets/t90_casebase_ph120.csv
- assets/t90_stage_policy.json
- assets/t90_casebase.parquet (optional, if present)
- assets/t90_casebase_ph120.parquet (optional, if present)
- assets/README.md (if present)
- core/__init__.py
- core/*.py runtime sources
- core/setup.py
- core/build_linux.sh
- core/*.so or core/*.pyd compiled extensions

This package is intended for CPU-only Linux deployment.
EOF

echo "[4/6] Verifying staged package imports"
(
  cd "${package_root}"
  "${PYTHON_BIN}" -c "from interface import recommend_t90_controls; print('package import check ok')"
)

echo "[5/6] Creating tar.gz archive"
tar -czf "${archive_path}" -C "${stage_dir}" T90

echo "[6/6] Delivery package created"
echo "stage_dir=${stage_dir}"
echo "archive_path=${archive_path}"

if [[ "${KEEP_STAGE}" -ne 1 ]]; then
  rm -rf "${stage_dir}"
  echo "staged directory removed after packaging"
fi
