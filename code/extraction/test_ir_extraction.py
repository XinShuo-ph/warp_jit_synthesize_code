#!/usr/bin/env python3
"""
Test cases for IR extraction - demonstrates Python → IR pairs
"""

import warp as wp
from ir_extractor import extract_kernel_ir_simple
import json

wp.init()

# Test Case 1: Simple arithmetic
@wp.kernel
def add_arrays(a: wp.array(dtype=float),
               b: wp.array(dtype=float),
               c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Test Case 2: Conditional logic
@wp.kernel
def clamp_values(x: wp.array(dtype=float),
                 min_val: float,
                 max_val: float,
                 y: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    y[tid] = val

# Test Case 3: Vector operations
@wp.kernel
def vec_dot_product(a: wp.array(dtype=wp.vec3),
                    b: wp.array(dtype=wp.vec3),
                    result: wp.array(dtype=float)):
    tid = wp.tid()
    result[tid] = wp.dot(a[tid], b[tid])

# Test Case 4: Loop with accumulation
@wp.kernel
def sum_neighbors(x: wp.array(dtype=float),
                  y: wp.array(dtype=float),
                  radius: int):
    tid = wp.tid()
    n = x.shape[0]
    total = float(0.0)  # Explicit dynamic variable
    count = int(0)      # Explicit dynamic variable
    
    for i in range(-radius, radius + 1):
        idx = tid + i
        if idx >= 0 and idx < n:
            total = total + x[idx]
            count = count + 1
    
    if count > 0:
        y[tid] = total / float(count)
    else:
        y[tid] = 0.0

# Test Case 5: Matrix-like indexing
@wp.kernel
def transpose_2d(input: wp.array2d(dtype=float),
                 output: wp.array2d(dtype=float)):
    i, j = wp.tid()
    output[j, i] = input[i, j]

def extract_and_save(kernel_func, test_name, *args, **kwargs):
    """Extract IR and save to file."""
    print(f"\n{'='*60}")
    print(f"Test Case: {test_name}")
    print(f"{'='*60}")
    
    # Extract IR
    ir = extract_kernel_ir_simple(kernel_func, *args, **kwargs)
    
    if ir:
        # Save to file
        output = {
            'test_name': test_name,
            'kernel_name': ir.kernel_name,
            'python_source': ir.python_source,
            'cpp_code': ir.cpp_code,
            'meta': ir.meta,
            'module_hash': ir.module_hash
        }
        
        filename = f"/workspace/data/{test_name}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✓ Saved to {filename}")
        print(f"  Python lines: {len(ir.python_source.splitlines())}")
        print(f"  C++ lines: {len(ir.cpp_code.splitlines())}")
        
        return True
    else:
        print(f"✗ Failed to extract IR")
        return False

def main():
    """Run all test cases."""
    
    # Test 1
    n = 10
    a = wp.array([float(i) for i in range(n)], dtype=float)
    b = wp.array([float(i*2) for i in range(n)], dtype=float)
    c = wp.zeros(n, dtype=float)
    extract_and_save(add_arrays, "test_01_add_arrays", dim=n, inputs=[a, b, c])
    
    # Test 2
    x = wp.array([float(i-5) for i in range(n)], dtype=float)
    y = wp.zeros(n, dtype=float)
    extract_and_save(clamp_values, "test_02_clamp_values", 
                    dim=n, inputs=[x, -2.0, 2.0, y])
    
    # Test 3
    vec_a = wp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=wp.vec3)
    vec_b = wp.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=wp.vec3)
    result = wp.zeros(3, dtype=float)
    extract_and_save(vec_dot_product, "test_03_vec_dot", 
                    dim=3, inputs=[vec_a, vec_b, result])
    
    # Test 4
    x = wp.array([float(i) for i in range(n)], dtype=float)
    y = wp.zeros(n, dtype=float)
    extract_and_save(sum_neighbors, "test_04_sum_neighbors",
                    dim=n, inputs=[x, y, 2])
    
    # Test 5
    m, n = 4, 5
    input_2d = wp.zeros((m, n), dtype=float)
    output_2d = wp.zeros((n, m), dtype=float)
    extract_and_save(transpose_2d, "test_05_transpose_2d",
                    dim=(m, n), inputs=[input_2d, output_2d])
    
    print("\n" + "="*60)
    print("All test cases completed!")
    print("="*60)

if __name__ == "__main__":
    main()
