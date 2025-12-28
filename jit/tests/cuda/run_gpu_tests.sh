#!/bin/bash
# Run CUDA GPU tests
# Execute this script on a machine with a GPU

set -e

echo "============================================"
echo "CUDA GPU Test Suite"
echo "============================================"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install warp-lang pytest numpy --quiet

# Check CUDA availability
echo ""
echo "Checking CUDA availability..."
python3 -c "
import warp as wp
wp.init()
print('CUDA available:', wp.is_cuda_available())
if wp.is_cuda_available():
    devices = wp.get_cuda_devices()
    for d in devices:
        print(f'  Device: {d}')
else:
    print('WARNING: No CUDA device found. GPU tests will be skipped.')
"

# Run extraction tests (no GPU required)
echo ""
echo "============================================"
echo "Running IR extraction tests (no GPU required)..."
echo "============================================"
cd "$(dirname "$0")/../.."
python3 -m pytest tests/cuda/test_extraction.py -v

# Run GPU tests
echo ""
echo "============================================"
echo "Running GPU execution tests (GPU required)..."
echo "============================================"
python3 -m pytest tests/cuda/test_kernels.py -v

echo ""
echo "============================================"
echo "All tests completed!"
echo "============================================"
