#!/usr/bin/env bash
set -euo pipefail

# resolve paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}/.."
BUILD_DIR="${ROOT_DIR}/build"
BINARY="${BUILD_DIR}/sgemm"

mkdir -p "${BUILD_DIR}"
echo "building sgemm... "

nvcc \
  "${SCRIPT_DIR}/host/main.cu" \
  "${SCRIPT_DIR}/kernels/sgemm_naive.cu" \
  -lcublas \
  -o "${BINARY}"

echo "build complete"

echo "running sgemm"
echo "---------------------"
"${BINARY}"
echo "---------------------"