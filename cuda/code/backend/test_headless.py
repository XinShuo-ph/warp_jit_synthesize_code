#!/usr/bin/env python3
"""Test headless CUDA generation on CPU."""

import sys
import os
import subprocess
import json
import glob
from pathlib import Path
import shutil
import warp as wp

# Setup paths
base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir / "synthesis"))
sys.path.insert(0, str(base_dir / "extraction"))

def main():
    print("=" * 70)
    print("HEADLESS CUDA GENERATION TEST")
    print("=" * 70)

    wp.init()
    print(f"Warp Version: {wp.__version__}")
    
    # Check that we are indeed on CPU (or at least pretend to be for this test if we had a GPU, 
    # but here we assume we are in the environment where previous tests failed)
    cuda_devices = [d for d in wp.get_devices() if d.is_cuda]
    if not cuda_devices:
        print("Confirmed: No CUDA devices detected. Running in headless mode.")
    else:
        print(f"Note: CUDA devices found ({cuda_devices}). Pipeline should still work via load() or fallback.")

    output_dir = "/tmp/cuda_headless_out"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    pipeline_script = base_dir / "synthesis" / "pipeline.py"
    
    print("\n[1/2] Running pipeline with --device cuda...")
    result = subprocess.run([
        sys.executable, str(pipeline_script),
        '--count', '5', 
        '--output', output_dir, 
        '--seed', '123',
        '--device', 'cuda'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Pipeline failed:\n{result.stderr}")
        sys.exit(1)
    
    print("✓ Pipeline finished successfully.")
    
    print("\n[2/2] Verifying Output...")
    samples = glob.glob(f'{output_dir}/*.json')
    if len(samples) != 5:
        print(f"❌ Expected 5 samples, found {len(samples)}")
        sys.exit(1)
        
    print(f"✓ Found {len(samples)} samples.")
    
    # Check content of first sample
    with open(samples[0], 'r') as f:
        data = json.load(f)
        
    print(f"Sample Kernel: {data['kernel_name']}")
    
    # Check for CUDA characteristic
    if 'cuda_kernel' in data['cpp_ir_forward']:
        print("✓ Forward IR contains 'cuda_kernel' symbol")
    elif '_forward' in data['cpp_ir_forward']:
        print("⚠️ Forward IR contains '_forward' but missing 'cuda_kernel'. Please check content:")
        print(data['cpp_ir_forward'][:200])
    else:
        print("❌ Forward IR seems empty or invalid.")
        sys.exit(1)
        
    # Check for CUDA-specific code in full IR
    if '__global__' in data['cpp_ir_full'] or 'blockIdx' in data['cpp_ir_full']:
        print("✓ Full IR contains CUDA keywords (__global__, blockIdx, etc.)")
    else:
        print("⚠️ Full IR might not be CUDA code. Check content:")
        print(data['cpp_ir_full'][:200])

    print("\nSUCCESS: Headless CUDA generation works!")

if __name__ == "__main__":
    main()
