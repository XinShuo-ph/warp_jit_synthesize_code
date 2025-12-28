#!/usr/bin/env python3
"""Test CUDA IR extraction for all kernel types."""
import sys
from pathlib import Path

# Add parent directories for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))
sys.path.insert(0, str(Path(__file__).parent))

import warp as wp
wp.init()

from generator import generate_kernel, GENERATORS
from pipeline import compile_kernel_from_source, extract_ir_from_kernel


def test_kernel_type(kernel_type: str, seed: int = 42):
    """Test CUDA extraction for a single kernel type."""
    print(f"\n{'='*60}")
    print(f"Testing: {kernel_type}")
    print('='*60)
    
    # Generate kernel
    spec = generate_kernel(kernel_type, seed=seed)
    print(f"Kernel: {spec.name}")
    print(f"Source:\n{spec.source}")
    
    # Compile kernel
    try:
        kernel = compile_kernel_from_source(spec.source, spec.name)
    except Exception as e:
        print(f"COMPILE FAILED: {e}")
        return False
    
    # Extract CPU IR
    try:
        cpu_ir = extract_ir_from_kernel(kernel, device="cpu")
        cpu_forward = cpu_ir["forward_code"]
        print(f"\n--- CPU Forward (snippet) ---")
        if cpu_forward:
            print(cpu_forward[:300] + "..." if len(cpu_forward) > 300 else cpu_forward)
        else:
            print("None")
    except Exception as e:
        print(f"CPU EXTRACTION FAILED: {e}")
        cpu_forward = None
    
    # Extract CUDA IR
    try:
        cuda_ir = extract_ir_from_kernel(kernel, device="cuda")
        cuda_forward = cuda_ir["forward_code"]
        print(f"\n--- CUDA Forward (snippet) ---")
        if cuda_forward:
            print(cuda_forward[:300] + "..." if len(cuda_forward) > 300 else cuda_forward)
        else:
            print("None")
    except Exception as e:
        print(f"CUDA EXTRACTION FAILED: {e}")
        cuda_forward = None
    
    # Verify differences
    if cpu_forward and cuda_forward:
        cpu_has_cpu = "_cpu_kernel_forward" in cpu_forward
        cuda_has_cuda = "_cuda_kernel_forward" in cuda_forward
        print(f"\n--- Verification ---")
        print(f"CPU has '_cpu_kernel_forward': {cpu_has_cpu}")
        print(f"CUDA has '_cuda_kernel_forward': {cuda_has_cuda}")
        
        if cpu_has_cpu and cuda_has_cuda:
            print("✓ PASS: Both devices generate correct IR")
            return True
        else:
            print("✗ FAIL: Device markers not found")
            return False
    else:
        print("✗ FAIL: Could not extract IR")
        return False


def main():
    """Test all kernel types."""
    print("CUDA IR Extraction Test Suite")
    print("="*60)
    
    results = {}
    for kernel_type in GENERATORS.keys():
        success = test_kernel_type(kernel_type)
        results[kernel_type] = success
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    for kernel_type, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {kernel_type}")
    
    return all(results.values())


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
