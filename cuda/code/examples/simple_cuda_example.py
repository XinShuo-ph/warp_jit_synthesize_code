#!/usr/bin/env python3
"""
Simple CUDA Code Generation Example

This example demonstrates generating CUDA code from Python kernels
without requiring a GPU. The code generation is pure Python.

For actual GPU execution, run on a machine with NVIDIA GPU.
"""
import sys
from pathlib import Path

# Add code paths
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

import warp as wp
wp.init()

from generator import KernelGenerator
from pipeline import compile_kernel_from_source, extract_ir_from_kernel


def main():
    print("=" * 60)
    print("CUDA Code Generation Example")
    print("=" * 60)
    
    # Create a kernel generator
    gen = KernelGenerator(seed=42)
    
    # Generate a simple arithmetic kernel
    spec = gen.generate("arithmetic")
    source = gen.to_python_source(spec)
    
    print("\n1. Python Kernel Source:")
    print("-" * 40)
    print(source)
    
    # Compile the kernel
    kernel = compile_kernel_from_source(source, spec.name)
    
    # Extract CUDA code (forward and backward)
    print("\n2. Extracting CUDA code...")
    ir = extract_ir_from_kernel(kernel, device="cuda", include_backward=True)
    
    print("\n3. CUDA Forward Kernel:")
    print("-" * 40)
    print(ir["forward_code"][:800] + "\n... [truncated]" if len(ir["forward_code"]) > 800 else ir["forward_code"])
    
    if ir["backward_code"]:
        print("\n4. CUDA Backward Kernel (first 500 chars):")
        print("-" * 40)
        print(ir["backward_code"][:500] + "\n... [truncated]")
    
    # Compare with CPU
    print("\n5. Comparing CPU vs CUDA:")
    print("-" * 40)
    cpu_ir = extract_ir_from_kernel(kernel, device="cpu", include_backward=False)
    
    print(f"CPU function name: ...{spec.name}..._cpu_kernel_forward")
    print(f"CUDA function name: ...{spec.name}..._cuda_kernel_forward")
    print(f"CPU code length: {len(cpu_ir['forward_code'])} chars")
    print(f"CUDA code length: {len(ir['forward_code'])} chars")
    
    # Show key CUDA-specific patterns
    cuda_patterns = [
        ("blockDim", "Thread block dimension"),
        ("blockIdx", "Block index"),
        ("threadIdx", "Thread index within block"),
        ("tile_shared_storage_t", "Shared memory"),
    ]
    
    print("\n6. CUDA-specific patterns found:")
    for pattern, description in cuda_patterns:
        found = pattern in ir["forward_code"]
        status = "✓" if found else "✗"
        print(f"  {status} {pattern}: {description}")
    
    print("\n" + "=" * 60)
    print("Done! CUDA code generated successfully (no GPU required).")
    print("=" * 60)


if __name__ == "__main__":
    main()
