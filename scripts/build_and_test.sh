#!/usr/bin/env bash
# build.sh - build, test, and optionally run the pipeline
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building imgpipeline ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j"$(nproc)"

echo ""
echo "=== Running Tests ==="
./test_pipeline

echo ""
echo "=== Running Benchmark (640x480, 500 frames) ==="
./benchmark_pipeline 640 480 500

echo ""
echo "=== Running Demo ==="
./imgpipeline_demo "${SCRIPT_DIR}/output"

echo ""
echo "Build complete. Binaries in ${BUILD_DIR}/"
