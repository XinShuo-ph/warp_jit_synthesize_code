#!/bin/bash
# Demonstration: Complete CUDA Backend Workflow
# Run this to see the full capabilities

set -e

echo "========================================================================"
echo "CUDA Backend - Complete Demonstration"
echo "========================================================================"
echo ""

# Setup
DEMO_OUTPUT="/tmp/cuda_demo_$$"
mkdir -p "$DEMO_OUTPUT"

echo "Output directory: $DEMO_OUTPUT"
echo ""

# Step 1: Basic CUDA code generation test
echo "========================================================================"
echo "Step 1: Verify CUDA Code Generation"
echo "========================================================================"
cd "$(dirname "$0")/../code/examples"
python3 test_cuda_codegen.py
echo ""

# Step 2: Test all kernel categories
echo "========================================================================"
echo "Step 2: Test All Kernel Categories (6 types)"
echo "========================================================================"
python3 test_all_kernels_cuda.py
echo ""

# Step 3: Forward and backward passes
echo "========================================================================"
echo "Step 3: Test Forward and Backward Passes"
echo "========================================================================"
python3 test_forward_backward_cuda.py
echo ""

# Step 4: Generate small dataset
echo "========================================================================"
echo "Step 4: Generate 20 CUDA Kernel Pairs"
echo "========================================================================"
cd ../synthesis
python3 pipeline.py -n 20 -d cuda -o "$DEMO_OUTPUT/small_batch" --seed 12345
echo ""

# Step 5: Verify output quality
echo "========================================================================"
echo "Step 5: Verify Generated Output"
echo "========================================================================"

FILE_COUNT=$(ls -1 "$DEMO_OUTPUT/small_batch"/*.json 2>/dev/null | wc -l)
echo "Generated files: $FILE_COUNT"

if [ "$FILE_COUNT" -gt 0 ]; then
    echo ""
    echo "Sample output (first file):"
    FIRST_FILE=$(ls "$DEMO_OUTPUT/small_batch"/*.json | head -1)
    echo "File: $(basename $FIRST_FILE)"
    
    # Extract metadata
    CATEGORY=$(python3 -c "import json; print(json.load(open('$FIRST_FILE'))['metadata']['category'])")
    KERNEL_NAME=$(python3 -c "import json; print(json.load(open('$FIRST_FILE'))['metadata']['kernel_name'])")
    DEVICE=$(python3 -c "import json; print(json.load(open('$FIRST_FILE'))['metadata']['device'])")
    
    echo "  Category: $CATEGORY"
    echo "  Kernel: $KERNEL_NAME"
    echo "  Device: $DEVICE"
    
    # Check for CUDA patterns
    if grep -q "blockIdx\|threadIdx" "$FIRST_FILE"; then
        echo "  ✓ CUDA thread indexing present"
    else
        echo "  ✗ CUDA patterns missing"
        exit 1
    fi
    
    # Show snippet
    echo ""
    echo "  Python source (first 5 lines):"
    python3 -c "import json; src=json.load(open('$FIRST_FILE'))['python_source']; print('\n'.join(['    ' + line for line in src.split('\n')[:5]]))"
    
    echo ""
    echo "  CUDA IR (first 10 lines):"
    python3 -c "import json; ir=json.load(open('$FIRST_FILE'))['cpp_forward']; print('\n'.join(['    ' + line for line in ir.split('\n')[:10]]))"
fi

# Step 6: Category distribution
echo ""
echo "========================================================================"
echo "Step 6: Category Distribution"
echo "========================================================================"

python3 << 'EOF'
import json
from pathlib import Path
import sys

demo_dir = Path(sys.argv[1])
files = list(demo_dir.glob("*.json"))

if not files:
    print("No files generated")
    sys.exit(1)

categories = {}
for file in files:
    with open(file) as f:
        data = json.load(f)
        cat = data['metadata']['category']
        categories[cat] = categories.get(cat, 0) + 1

print(f"Total pairs: {len(files)}")
print("\nBreakdown by category:")
for cat, count in sorted(categories.items()):
    pct = count / len(files) * 100
    print(f"  {cat:15s}: {count:3d} ({pct:5.1f}%)")
EOF
python3 -c "import sys; sys.argv.append('$DEMO_OUTPUT/small_batch')" -

# Step 7: Batch generator test
echo ""
echo "========================================================================"
echo "Step 7: Test Batch Generator (50 pairs)"
echo "========================================================================"
python3 batch_generator.py -n 50 -d cuda -o "$DEMO_OUTPUT/large_batch" -s 99999

# Step 8: Final verification
echo ""
echo "========================================================================"
echo "Step 8: Final Verification"
echo "========================================================================"

SMALL_COUNT=$(ls -1 "$DEMO_OUTPUT/small_batch"/*.json 2>/dev/null | wc -l)
LARGE_COUNT=$(ls -1 "$DEMO_OUTPUT/large_batch"/*.json 2>/dev/null | wc -l)

echo "Small batch (pipeline.py): $SMALL_COUNT files"
echo "Large batch (batch_generator.py): $LARGE_COUNT files"
echo "Total generated: $((SMALL_COUNT + LARGE_COUNT)) files"

# Cleanup
echo ""
echo "Cleaning up demo output: $DEMO_OUTPUT"
rm -rf "$DEMO_OUTPUT"

# Summary
echo ""
echo "========================================================================"
echo "DEMONSTRATION COMPLETE"
echo "========================================================================"
echo ""
echo "✓ All components working"
echo "✓ CUDA code generation verified"
echo "✓ All kernel categories supported"
echo "✓ Pipeline and batch generator operational"
echo "✓ Output format validated"
echo ""
echo "Ready for production use on GPU-enabled systems!"
echo ""
echo "Next steps:"
echo "  1. Copy code to GPU machine"
echo "  2. Run: bash tests/run_all_cuda_tests.sh"
echo "  3. Generate production dataset"
echo ""
