"""Test cases for IR extraction - diverse kernel patterns."""

import warp as wp
import sys
import json
import os
sys.path.insert(0, '/workspace/code')

from extraction.ir_extractor import extract_ir_pair

wp.init()

# Test Case 1: Simple arithmetic
@wp.kernel
def arithmetic_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    result[tid] = a[tid] * 2.0 + 1.0 - 0.5

# Test Case 2: Array indexing
@wp.kernel
def indexing_kernel(data: wp.array(dtype=float), indices: wp.array(dtype=int), result: wp.array(dtype=float)):
    tid = wp.tid()
    idx = indices[tid]
    result[tid] = data[idx]

# Test Case 3: Conditional (if statement)
@wp.kernel
def conditional_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    val = a[tid]
    if val > 0.0:
        result[tid] = val * 2.0
    else:
        result[tid] = 0.0

# Test Case 4: Loop
@wp.kernel
def loop_kernel(a: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    sum_val = 0.0
    for i in range(3):
        sum_val = sum_val + a[tid]
    result[tid] = sum_val

# Test Case 5: Vector operations
@wp.kernel
def vector_kernel(a: wp.array(dtype=wp.vec3), result: wp.array(dtype=float)):
    tid = wp.tid()
    v = a[tid]
    result[tid] = wp.length(v)

# Test Case 6: Multiple operations
@wp.kernel
def complex_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    x = a[tid]
    y = b[tid]
    temp = x * x + y * y
    c[tid] = wp.sqrt(temp)

print("Compiling test kernels and extracting IR...")
print("=" * 70)

test_cases = []

# Compile and extract each kernel
kernels = [
    ("arithmetic", arithmetic_kernel, [wp.array([1.0, 2.0, 3.0], dtype=float), wp.zeros(3, dtype=float)]),
    ("indexing", indexing_kernel, [
        wp.array([10.0, 20.0, 30.0, 40.0], dtype=float),
        wp.array([0, 2, 1], dtype=int),
        wp.zeros(3, dtype=float)
    ]),
    ("conditional", conditional_kernel, [
        wp.array([-1.0, 2.0, -3.0, 4.0], dtype=float),
        wp.zeros(4, dtype=float)
    ]),
    ("loop", loop_kernel, [
        wp.array([1.0, 2.0, 3.0], dtype=float),
        wp.zeros(3, dtype=float)
    ]),
    ("vector", vector_kernel, [
        wp.array([wp.vec3(1.0, 2.0, 3.0), wp.vec3(4.0, 5.0, 6.0)], dtype=wp.vec3),
        wp.zeros(2, dtype=float)
    ]),
    ("complex", complex_kernel, [
        wp.array([3.0, 5.0], dtype=float),
        wp.array([4.0, 12.0], dtype=float),
        wp.zeros(2, dtype=float)
    ]),
]

for name, kernel, inputs in kernels:
    # Compile the kernel
    dim = len(inputs[-1])  # Result array is always last
    wp.launch(kernel, dim=dim, inputs=inputs)
    wp.synchronize()
    
    # Extract IR
    python_src, cpp_ir = extract_ir_pair(kernel)
    
    if cpp_ir:
        test_cases.append({
            'name': name,
            'python_source': python_src,
            'cpp_ir': cpp_ir,
            'kernel_name': kernel.func.__name__,
        })
        print(f"✓ {name}: {len(python_src)} chars Python → {len(cpp_ir)} chars C++")
    else:
        print(f"✗ {name}: FAILED to extract IR")

print("\n" + "=" * 70)
print(f"Successfully extracted {len(test_cases)} test cases")

# Save to JSON
output_dir = '/workspace/data'
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, 'test_cases.json')

with open(output_file, 'w') as f:
    json.dump(test_cases, f, indent=2)

print(f"Saved test cases to: {output_file}")

# Show first example
if test_cases:
    print("\n" + "=" * 70)
    print(f"Example: {test_cases[0]['name']}")
    print("=" * 70)
    print("Python:")
    print(test_cases[0]['python_source'])
    print("\nC++ IR (first 500 chars):")
    print(test_cases[0]['cpp_ir'][:500])
