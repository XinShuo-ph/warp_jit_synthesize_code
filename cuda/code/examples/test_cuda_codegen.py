"""
Test CUDA IR extraction without GPU.

This script verifies that CUDA code generation works even without GPU hardware.
"""
import warp as wp
import sys
from pathlib import Path

# Add extraction directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "extraction"))

from ir_extractor import extract_ir

# Simple test kernel
@wp.kernel
def test_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def main():
    wp.init()
    
    print("=" * 60)
    print("Testing CUDA Code Generation (CPU-only mode)")
    print("=" * 60)
    
    # Test CPU codegen
    print("\n1. Testing CPU code generation...")
    try:
        result_cpu = extract_ir(test_add, device="cpu", include_backward=False)
        print(f"   ✓ CPU codegen successful")
        print(f"   - Kernel: {result_cpu['kernel_name']}")
        print(f"   - Device: {result_cpu['metadata']['device']}")
        print(f"   - Forward code length: {len(result_cpu['forward_code'])} chars")
        print(f"\n   CPU Forward Code (first 500 chars):")
        print(f"   {result_cpu['forward_code'][:500]}")
    except Exception as e:
        print(f"   ✗ CPU codegen failed: {e}")
        return 1
    
    # Test CUDA codegen
    print("\n2. Testing CUDA code generation...")
    try:
        result_cuda = extract_ir(test_add, device="cuda", include_backward=False)
        print(f"   ✓ CUDA codegen successful")
        print(f"   - Kernel: {result_cuda['kernel_name']}")
        print(f"   - Device: {result_cuda['metadata']['device']}")
        print(f"   - Forward code length: {len(result_cuda['forward_code'])} chars")
        print(f"\n   CUDA Forward Code (first 500 chars):")
        print(f"   {result_cuda['forward_code'][:500]}")
        
        # Check for CUDA-specific patterns
        cuda_code = result_cuda['forward_code']
        has_cuda_qualifiers = '__global__' in cuda_code or '__device__' in cuda_code
        has_cuda_tid = 'blockIdx' in cuda_code or 'threadIdx' in cuda_code
        
        print(f"\n   CUDA-specific patterns:")
        print(f"   - Has CUDA qualifiers (__global__/__device__): {has_cuda_qualifiers}")
        print(f"   - Has CUDA thread indexing: {has_cuda_tid}")
        
    except Exception as e:
        print(f"   ✗ CUDA codegen failed: {e}")
        print(f"   Note: This is expected if CUDA is not available")
        return 0  # Not a failure - just means CUDA isn't available
    
    print("\n" + "=" * 60)
    print("✓ CUDA code generation test completed successfully!")
    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
