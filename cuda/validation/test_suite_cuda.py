#!/usr/bin/env python3
"""Comprehensive test of CUDA backend."""

import sys
import os
import subprocess
import json
import glob
from pathlib import Path
import shutil

# Setup paths
base_dir = Path(__file__).resolve().parent.parent / "code" / "backend"
sys.path.insert(0, str(base_dir / "synthesis"))
sys.path.insert(0, str(base_dir / "extraction"))

print("=" * 70)
print("CUDA BACKEND VALIDATION TEST")
print("=" * 70)

# Check Warp and CUDA
import warp as wp
wp.init()
if not wp.is_cuda_available():
    print("WARNING: No CUDA device found. Tests will fail or be skipped.")
    has_cuda = False
else:
    try:
        print(f"CUDA Device Found: {wp.get_device('cuda')}")
        has_cuda = True
    except Exception:
         print("WARNING: CUDA available but get_device('cuda') failed.")
         has_cuda = False

# Test 1: Generator (Platform independent)
print("\n[1/3] Testing generator...")
try:
    from generator import KernelGenerator
    gen = KernelGenerator()
    methods = [m for m in dir(gen) if m.startswith('gen_')]
    print(f"   ✓ Found {len(methods)} generator methods")
except ImportError as e:
    print(f"   ❌ Failed to import generator: {e}")

# Test 2: Pipeline with CUDA
print("\n[2/3] Testing pipeline generation (CUDA)...")
if has_cuda:
    output_dir = "/tmp/cuda_test_out"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    pipeline_script = base_dir / "synthesis" / "pipeline.py"
    
    result = subprocess.run([
        sys.executable, str(pipeline_script),
        '--count', '5', 
        '--output', output_dir, 
        '--seed', '999',
        '--device', 'cuda'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"   ❌ Pipeline failed:\n{result.stderr}")
    else:
        print("   ✓ Pipeline generated samples successfully")
        
        # Verify output
        samples = glob.glob(f'{output_dir}/*.json')
        if len(samples) >= 5:
            print(f"   ✓ Found {len(samples)} samples")
            try:
                sample = json.load(open(samples[0]))
                if sample['kernel_type']:
                    print(f"   ✓ Sample type: {sample['kernel_type']}")
                
                # Check for CUDA characteristic
                if 'cuda_kernel' in sample['cpp_ir_forward']:
                     print("   ✓ IR contains 'cuda_kernel' symbol")
                elif '_forward' in sample['cpp_ir_forward']:
                     print("   ✓ IR contains '_forward' symbol (Naming convention verification needed)")
                else:
                     print("   ⚠️ IR seems empty or missing symbols")
            except Exception as e:
                print(f"   ❌ Failed to parse sample: {e}")
        else:
            print(f"   ❌ Expected 5 samples, found {len(samples)}")
else:
    print("   ⚠️ Skipped (No CUDA Device)")

# Test 3: Kernel Validation
print("\n[3/3] Examples Validation")
print("   To verify examples on CUDA, use the 'examples' scripts with a CUDA-enabled warp context.")
print("   (This is pending M3/M4 tasks to port specific examples if needed)")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
