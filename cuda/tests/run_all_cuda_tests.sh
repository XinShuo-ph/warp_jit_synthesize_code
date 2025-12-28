#!/bin/bash
# Master test script for CUDA backend validation
#
# Run this on GPU machine:
#   bash run_all_cuda_tests.sh
#
# Expected: All tests pass

set -e

echo "========================================================================"
echo "CUDA Backend Validation Test Suite"
echo "========================================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "✗ python3 not found"
    exit 1
fi
echo "✓ python3 found: $(python3 --version)"

# Check warp
if ! python3 -c "import warp" 2>/dev/null; then
    echo "✗ warp-lang not installed"
    echo "  Install with: pip install warp-lang"
    exit 1
fi
echo "✓ warp-lang installed"

# Check CUDA
echo ""
echo "Checking CUDA availability..."
python3 -c "import warp as wp; wp.init(); print('CUDA devices:', list(wp.get_devices()) if hasattr(wp, 'get_devices') else 'N/A')" || true

echo ""
echo "========================================================================"
echo "Test 1: Basic CUDA Code Generation"
echo "========================================================================"
cd "$(dirname "$0")/../code/examples"
python3 test_cuda_codegen.py

echo ""
echo "========================================================================"
echo "Test 2: All Kernel Categories"
echo "========================================================================"
python3 test_all_kernels_cuda.py

echo ""
echo "========================================================================"
echo "Test 3: Forward and Backward Passes"
echo "========================================================================"
python3 test_forward_backward_cuda.py

echo ""
echo "========================================================================"
echo "Test 4: Pipeline End-to-End"
echo "========================================================================"
cd "$(dirname "$0")"
python3 test_cuda_pipeline.py

echo ""
echo "========================================================================"
echo "Test 5: Generate Sample Dataset (10 pairs)"
echo "========================================================================"
cd "$(dirname "$0")/../code/synthesis"
OUTPUT_DIR="/tmp/cuda_test_output_$$"
python3 pipeline.py -n 10 -d cuda -o "$OUTPUT_DIR" --seed 42

# Verify output
if [ -d "$OUTPUT_DIR" ] && [ "$(ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)" -gt 0 ]; then
    COUNT=$(ls -1 "$OUTPUT_DIR"/*.json | wc -l)
    echo "✓ Generated $COUNT JSON files"
    
    # Check first file has CUDA code
    FIRST_FILE=$(ls "$OUTPUT_DIR"/*.json | head -1)
    if grep -q "blockIdx\|threadIdx" "$FIRST_FILE"; then
        echo "✓ CUDA thread indexing present in output"
    else
        echo "✗ CUDA patterns not found in output"
        exit 1
    fi
    
    rm -rf "$OUTPUT_DIR"
else
    echo "✗ Output directory empty or missing"
    exit 1
fi

echo ""
echo "========================================================================"
echo "ALL TESTS PASSED"
echo "========================================================================"
echo ""
echo "CUDA backend is fully functional!"
echo ""
echo "Next steps:"
echo "  - Generate large dataset: python3 batch_generator.py -n 10000 -d cuda -o /output/dir"
echo "  - Use in training pipeline"
echo ""
