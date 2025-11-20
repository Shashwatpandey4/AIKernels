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
  -O3\
  -arch=sm_89\
  "${SCRIPT_DIR}/host/main.cu" \
  "${SCRIPT_DIR}/kernels/sgemm.cu" \
  -o "${BINARY}"

echo "build complete"
echo

echo "benchmarking"
echo "-----------------------------------------------------"
sum=0
count=0

for i in {1..10}; do
    output=$("${BINARY}")

    gflops=$(echo "$output" | awk '{print $(NF-1)}' | tail -n1)
    echo "Run $i: $gflops GFLOP/s"

    if (( i > 1 )); then
        sum=$(echo "$sum + $gflops" | bc -l)
        count=$((count+1))
    fi
done

avg=$(echo "scale=3; $sum / $count" | bc -l)
echo "-----------------------------------------------------"
echo "Average : $avg GFLOP/s"
