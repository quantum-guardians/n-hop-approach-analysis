#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Usage:
#   ./run_face_k_analysis_extended.sh [output_dir]
#
# Optional environment overrides:
#   SEED=42
#   NUM_GRAPHS=20
#   NUM_SAMPLES=1
#   PYTHON_BIN=python
#   MPLCONFIGDIR=/custom/writable/path
#
# Notes:
# - This script is intentionally sized for a long-running broad sweep.
# - It covers larger n, coarser removal ratios, and a wider target_k spectrum.
# - NUM_SAMPLES is still passed through for CLI compatibility.

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
NUM_GRAPHS="${NUM_GRAPHS:-20}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
OUTPUT_DIR="${1:-results/face_k_analysis_extended}"

SIZES=(
  20 40 60 80 100 150 200 300 400
)

REMOVAL_PCTS=(
  0.0 0.1 0.2 0.3 0.4 0.5
)

TARGET_KS=(
  1 2 3 4 5 6 7 8 9 10
  11 12 13 14 15 16 17 18 19 20
  21 22 23 24 25 26 27 28 29 30
  31 32 33 34 35 36 37 38 39 40
  41 42 43 44 45 46 47 48 49 50
)

echo "Starting extended face-k sweep"
echo "  output dir   : $OUTPUT_DIR"
echo "  seed         : $SEED"
echo "  num graphs   : $NUM_GRAPHS"
echo "  num samples  : $NUM_SAMPLES"
echo "  sizes        : ${SIZES[*]}"
echo "  removal pcts : ${REMOVAL_PCTS[*]}"
echo "  target ks    : ${TARGET_KS[*]}"

mkdir -p "$OUTPUT_DIR"
DEFAULT_MPLCONFIGDIR="$ROOT_DIR/.cache/matplotlib"
mkdir -p "$DEFAULT_MPLCONFIGDIR"

MPLCONFIGDIR="${MPLCONFIGDIR:-$DEFAULT_MPLCONFIGDIR}" \
"$PYTHON_BIN" main.py face-k-analysis \
  --sizes "${SIZES[@]}" \
  --removal-pcts "${REMOVAL_PCTS[@]}" \
  --target-ks "${TARGET_KS[@]}" \
  --num-graphs "$NUM_GRAPHS" \
  --num-samples "$NUM_SAMPLES" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"

echo "Extended face-k sweep completed."
