"""Test the IR extractor with a simple kernel."""

import warp as wp
import sys
sys.path.insert(0, '/workspace/code')

from extraction.ir_extractor import extract_ir, extract_ir_pair

# Initialize warp
wp.init()

# Define test kernel
@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    """Add two arrays element-wise."""
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Manually trigger compilation with real data
n = 5
a = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
b = wp.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
c = wp.zeros(n, dtype=float)

print("Compiling kernel...")
wp.launch(add_kernel, dim=n, inputs=[a, b, c])
wp.synchronize()
print(f"Result: {c.numpy()}")

# Test the extractor
print("\nTesting IR extraction...")
print("=" * 70)

# Extract IR (don't trigger compilation again)
result = extract_ir(add_kernel, trigger_compile=False)

print(f"Kernel name: {result['kernel_name']}")
print(f"Module name: {result['module_name']}")
print(f"Signature: {result['signature']}")
print(f"Cache path: {result['cache_path']}")

print("\n" + "=" * 70)
print("Python Source:")
print("=" * 70)
print(result['python_source'])

print("\n" + "=" * 70)
print("C++ Kernel IR:")
print("=" * 70)
if result['cpp_kernel']:
    print(result['cpp_kernel'][:800])
    print("...")
    print(f"\n(Total length: {len(result['cpp_kernel'])} chars)")
else:
    print("ERROR: No C++ IR found!")

print("\n" + "=" * 70)
print("Testing extract_ir_pair()...")
python_src, cpp_ir = extract_ir_pair(add_kernel)
print(f"Python source length: {len(python_src)} chars")
print(f"C++ IR length: {len(cpp_ir) if cpp_ir else 0} chars")
print("SUCCESS!" if cpp_ir else "FAILED!")
