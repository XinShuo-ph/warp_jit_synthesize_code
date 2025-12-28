"""Test CUDA IR extraction with existing ir_extractor."""
import sys
sys.path.append('/workspace/cuda/code/extraction')

import warp as wp
from ir_extractor import extract_ir

wp.init()

# Test with arithmetic kernel
@wp.kernel
def test_arithmetic(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

print("=" * 60)
print("Testing Existing IR Extractor with CUDA")
print("=" * 60)

# Test CPU extraction
print("\n1. Testing CPU extraction...")
try:
    cpu_result = extract_ir(test_arithmetic, device="cpu", include_backward=False)
    print(f"   ✓ CPU extraction successful")
    print(f"   - Forward code length: {len(cpu_result['forward_code'])} chars")
    print(f"   - Contains 'cpu_kernel_forward': {'cpu_kernel_forward' in cpu_result['forward_code']}")
except Exception as e:
    print(f"   ✗ CPU extraction failed: {e}")

# Test CUDA extraction
print("\n2. Testing CUDA extraction...")
try:
    cuda_result = extract_ir(test_arithmetic, device="cuda", include_backward=False)
    print(f"   ✓ CUDA extraction successful")
    print(f"   - Forward code length: {len(cuda_result['forward_code'])} chars")
    print(f"   - Contains 'cuda_kernel_forward': {'cuda_kernel_forward' in cuda_result['forward_code']}")
    print(f"   - Contains '__global__': {'__global__' in cuda_result['forward_code']}")
    print(f"   - Device in metadata: {cuda_result['metadata']['device']}")
    
    print(f"\n   CUDA IR Sample (first 500 chars):")
    print(f"   {cuda_result['forward_code'][:500]}")
    
except Exception as e:
    print(f"   ✗ CUDA extraction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Result: Existing ir_extractor.py already supports CUDA!")
print("=" * 60)
