#!/bin/bash
# GPU Test Runner Script
# Run this script on a machine with NVIDIA GPU and CUDA driver

set -e

echo "========================================"
echo "  CUDA Backend Validation Script"
echo "========================================"
echo ""

# Navigate to jit directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JIT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$JIT_DIR"

echo "Working directory: $(pwd)"
echo ""

# Step 1: Check Python and warp
echo "Step 1: Checking environment..."
echo "----------------------------------------"
python3 --version

# Check warp installation
python3 -c "import warp as wp; wp.init(); print(f'Warp version: {wp.__version__}')"

# Check CUDA availability
echo ""
echo "Checking CUDA devices..."
python3 -c "
import warp as wp
wp.init()
devices = wp.get_devices()
print(f'Available devices: {devices}')
cuda_available = any('cuda' in str(d) for d in devices)
if cuda_available:
    print('✓ CUDA is available')
else:
    print('✗ CUDA is NOT available')
    print('Please run on a machine with NVIDIA GPU and CUDA driver')
    exit(1)
"

echo ""
echo "Step 2: Testing IR Extractor with CUDA..."
echo "----------------------------------------"
python3 code/extraction/ir_extractor.py --device cuda

echo ""
echo "Step 3: Testing Pipeline with CUDA..."
echo "----------------------------------------"
rm -rf data/cuda_validation
python3 code/synthesis/pipeline.py --device cuda --count 10 --output data/cuda_validation

echo ""
echo "Step 4: Validating Generated Data..."
echo "----------------------------------------"
python3 -c "
import json
from pathlib import Path

data_dir = Path('data/cuda_validation')
files = list(data_dir.glob('*.json'))
print(f'Generated files: {len(files)}')

for f in files[:3]:
    with open(f) as fp:
        data = json.load(fp)
    print(f'  {f.name}:')
    print(f'    - kernel_type: {data[\"kernel_type\"]}')
    print(f'    - device: {data[\"device\"]}')
    print(f'    - ir_forward length: {len(data[\"ir_forward\"])} chars')
    
    # Check for CUDA patterns
    if 'cuda_kernel_forward' in data['ir_forward'] or '__global__' in data['ir_forward']:
        print(f'    - ✓ Contains CUDA patterns')
    else:
        print(f'    - ✗ Missing CUDA patterns')
"

echo ""
echo "Step 5: Running pytest tests..."
echo "----------------------------------------"
python3 -m pytest tests/test_cuda_kernels.py -v

echo ""
echo "========================================"
echo "  All GPU Tests Completed Successfully!"
echo "========================================"
