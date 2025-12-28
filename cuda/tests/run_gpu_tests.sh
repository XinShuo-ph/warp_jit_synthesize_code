#!/bin/bash
# GPU Test Runner for CUDA Backend
# Run this script on a system with CUDA-capable GPU

set -e

echo "======================================================================"
echo "CUDA Backend GPU Test Suite"
echo "======================================================================"
echo ""
echo "This script runs tests on CUDA-capable GPU."
echo "Requirements:"
echo "  - CUDA-capable GPU"
echo "  - NVIDIA drivers installed"
echo "  - warp-lang package installed"
echo ""

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ nvidia-smi not found!"
    echo "  Please install NVIDIA drivers"
    exit 1
fi

echo "======================================================================"
echo "GPU Information"
echo "======================================================================"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ python3 not found!"
    exit 1
fi

echo "======================================================================"
echo "Python Environment"
echo "======================================================================"
python3 --version
echo ""

# Check warp installation
echo "Checking warp-lang installation..."
if python3 -c "import warp" 2>/dev/null; then
    python3 -c "import warp as wp; wp.init(); print(f'Warp version: {wp.__version__}')"
    echo "✓ warp-lang installed"
else
    echo "✗ warp-lang not installed!"
    echo "  Install with: pip install warp-lang"
    exit 1
fi
echo ""

# Change to cuda directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR/.."

echo "======================================================================"
echo "Test 1: CUDA Code Structure Validation (CPU-only, no GPU needed)"
echo "======================================================================"
python3 tests/test_cuda_kernels.py
TEST1_RESULT=$?
echo ""

if [ $TEST1_RESULT -ne 0 ]; then
    echo "✗ Structure tests failed!"
    exit 1
fi

echo "======================================================================"
echo "Test 2: GPU Execution Tests (requires GPU)"
echo "======================================================================"
python3 tests/run_on_gpu.py
TEST2_RESULT=$?
echo ""

if [ $TEST2_RESULT -ne 0 ]; then
    echo "✗ GPU execution tests failed!"
    exit 1
fi

echo "======================================================================"
echo "✓ ALL TESTS PASSED"
echo "======================================================================"
echo ""
echo "Your CUDA backend is working correctly!"
echo ""
echo "Next steps:"
echo "  1. Generate large dataset: python3 code/synthesis/cuda_batch_generator.py -n 1000"
echo "  2. Explore samples: ls -lh data/cuda_large/"
echo "  3. View sample: python3 -c \"import json; print(json.dumps(json.load(open('data/cuda_large/cuda_synth_0000.json')), indent=2))\""
echo ""

exit 0
