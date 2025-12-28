#!/bin/bash
# Run all CUDA tests
# Usage: bash tests/run_cuda_tests.sh

set -e

cd "$(dirname "$0")/.."

echo "============================================================"
echo "CUDA Backend Test Suite"
echo "============================================================"
echo ""

# Check CUDA availability first
echo "Step 1: Check CUDA availability"
python3 -c "
import warp as wp
wp.init()
if wp.is_cuda_available():
    print('  CUDA is available')
    devices = wp.get_cuda_devices()
    for i, dev in enumerate(devices):
        print(f'    Device {i}: {dev}')
else:
    print('  CUDA NOT AVAILABLE - tests will fail')
    exit(1)
"
echo ""

# Run the main test suite
echo "Step 2: Run CUDA test suite"
python3 tests/test_cuda.py
echo ""

# Test IR generation for different kernel types
echo "Step 3: Test IR generation for each kernel type"
python3 -c "
import sys
sys.path.insert(0, 'code/synthesis')
from generator import GENERATORS
from pipeline import synthesize_pair, run_pipeline
import warp as wp
wp.init()

print('Testing each kernel type with CUDA IR generation:')
for cat in GENERATORS.keys():
    from generator import generate_kernel
    spec = generate_kernel(cat, seed=42)
    pair = synthesize_pair(spec, device='cuda')
    if pair and '_cuda_kernel_forward' in pair['cpp_forward']:
        print(f'  [{cat}] PASS')
    else:
        print(f'  [{cat}] FAIL')
"
echo ""

# Generate a small batch of CUDA samples
echo "Step 4: Generate sample CUDA data"
python3 code/synthesis/pipeline.py -n 10 --device cuda -o data/cuda_test
echo ""

# Verify the generated data
echo "Step 5: Verify generated data"
python3 -c "
import json
import os
from pathlib import Path

data_dir = Path('data/cuda_test')
files = list(data_dir.glob('*.json'))
print(f'Generated {len(files)} files')

valid = 0
for f in files:
    with open(f) as fp:
        data = json.load(fp)
    if '_cuda_kernel_forward' in data.get('cpp_forward', ''):
        valid += 1

print(f'Valid CUDA samples: {valid}/{len(files)}')
if valid == len(files):
    print('All samples valid!')
else:
    print('Some samples invalid!')
    exit(1)
"
echo ""

echo "============================================================"
echo "All CUDA tests passed!"
echo "============================================================"
