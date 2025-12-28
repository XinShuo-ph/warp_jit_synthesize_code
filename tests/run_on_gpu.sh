#!/bin/bash
# Script to run CUDA tests on GPU hardware
# Usage: ./run_on_gpu.sh

set -e

echo "=================================================="
echo "CUDA Kernel Validation Script"
echo "=================================================="
echo ""

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    echo "   This script requires NVIDIA GPU with CUDA support"
    exit 1
fi

echo "✓ nvidia-smi found"
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Check Python and warp
echo "Checking Python environment..."
python3 --version
echo ""

if ! python3 -c "import warp" 2>/dev/null; then
    echo "❌ Warp not installed"
    echo "   Installing warp-lang..."
    pip install warp-lang
fi

echo "✓ Warp installed"
echo ""

# Run tests
echo "=================================================="
echo "Running CUDA Kernel Tests"
echo "=================================================="
echo ""

cd "$(dirname "$0")/.."
python3 tests/test_cuda_kernels.py

TEST_RESULT=$?

echo ""
echo "=================================================="
if [ $TEST_RESULT -eq 0 ]; then
    echo "✅ All tests passed!"
else
    echo "❌ Some tests failed (exit code: $TEST_RESULT)"
fi
echo "=================================================="

exit $TEST_RESULT
