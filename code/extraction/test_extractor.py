"""
Test IR Extractor

This script tests the IR extraction functionality on various kernel types.
"""

import sys
sys.path.insert(0, '/workspace/code')

import warp as wp
import numpy as np
from extraction.ir_extractor import extract_ir, extract_ir_to_file

wp.init()


# Test Case 1: Simple arithmetic
@wp.kernel
def test_arithmetic(a: wp.array(dtype=float), 
                    b: wp.array(dtype=float),
                    c: wp.array(dtype=float)):
    """Simple element-wise operations."""
    i = wp.tid()
    c[i] = a[i] * 2.0 + b[i] - 1.0


# Test Case 2: Vector operations
@wp.kernel
def test_vectors(positions: wp.array(dtype=wp.vec3),
                 velocities: wp.array(dtype=wp.vec3),
                 forces: wp.array(dtype=wp.vec3),
                 dt: float):
    """Vector math and built-in functions."""
    i = wp.tid()
    
    vel = velocities[i]
    force = forces[i]
    
    # Update velocity
    new_vel = vel + force * dt
    
    # Clamp velocity magnitude
    speed = wp.length(new_vel)
    if speed > 10.0:
        new_vel = wp.normalize(new_vel) * 10.0
    
    velocities[i] = new_vel


# Test Case 3: Control flow
@wp.kernel
def test_control_flow(data: wp.array(dtype=float),
                      threshold: float,
                      output: wp.array(dtype=float)):
    """Conditional branching."""
    i = wp.tid()
    
    val = data[i]
    
    if val < 0.0:
        output[i] = -val
    elif val < threshold:
        output[i] = val * 2.0
    else:
        output[i] = threshold + (val - threshold) * 0.5


# Test Case 4: Loops
@wp.kernel
def test_loops(matrix: wp.array(dtype=float, ndim=2),
               vector: wp.array(dtype=float),
               result: wp.array(dtype=float),
               n: int):
    """Matrix-vector multiplication with explicit loop."""
    i = wp.tid()
    
    sum_val = float(0.0)  # Must use float() for mutable loop variable
    for j in range(n):
        sum_val = sum_val + matrix[i, j] * vector[j]
    
    result[i] = sum_val


# Test Case 5: Functions
@wp.func
def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    return 1.0 / (1.0 + wp.exp(-x))


@wp.kernel
def test_functions(inputs: wp.array(dtype=float),
                   weights: wp.array(dtype=float),
                   bias: float,
                   outputs: wp.array(dtype=float)):
    """Kernel using helper function."""
    i = wp.tid()
    
    # Weighted sum
    z = inputs[i] * weights[i] + bias
    
    # Apply activation
    outputs[i] = sigmoid(z)


def main():
    print("=" * 60)
    print("IR EXTRACTOR TEST SUITE")
    print("=" * 60)
    print()
    
    # Compile all kernels by running them once
    n = 10
    
    # Test 1: Arithmetic
    a = wp.array(np.ones(n), dtype=float)
    b = wp.array(np.ones(n), dtype=float)
    c = wp.zeros(n, dtype=float)
    wp.launch(test_arithmetic, dim=n, inputs=[a, b, c])
    
    # Test 2: Vectors
    pos = wp.zeros(n, dtype=wp.vec3)
    vel = wp.zeros(n, dtype=wp.vec3)
    forces = wp.zeros(n, dtype=wp.vec3)
    wp.launch(test_vectors, dim=n, inputs=[pos, vel, forces, 0.1])
    
    # Test 3: Control flow
    data = wp.array(np.linspace(-5, 5, n), dtype=float)
    output = wp.zeros(n, dtype=float)
    wp.launch(test_control_flow, dim=n, inputs=[data, 2.0, output])
    
    # Test 4: Loops
    matrix = wp.zeros((5, 5), dtype=float)
    vector = wp.ones(5, dtype=float)
    result = wp.zeros(5, dtype=float)
    wp.launch(test_loops, dim=5, inputs=[matrix, vector, result, 5])
    
    # Test 5: Functions
    inputs = wp.ones(n, dtype=float)
    weights = wp.ones(n, dtype=float)
    outputs = wp.zeros(n, dtype=float)
    wp.launch(test_functions, dim=n, inputs=[inputs, weights, 0.5, outputs])
    
    wp.synchronize()
    
    print("All kernels compiled successfully!")
    print()
    
    # Extract IR for all test cases
    test_kernels = [
        test_arithmetic,
        test_vectors,
        test_control_flow,
        test_loops,
        test_functions
    ]
    
    output_dir = "/workspace/data/test_cases"
    
    print("Extracting IR...")
    print()
    
    success_count = 0
    for kernel in test_kernels:
        if extract_ir_to_file(kernel, output_dir):
            success_count += 1
        print()
    
    print("=" * 60)
    print(f"EXTRACTION COMPLETE: {success_count}/{len(test_kernels)} successful")
    print("=" * 60)
    
    # Test extraction API directly
    print("\nTesting direct extraction API...")
    result = extract_ir(test_arithmetic)
    
    if result.success:
        print(f"✓ Direct extraction successful")
        print(f"  Kernel: {result.kernel_name}")
        print(f"  Module: {result.module_name}")
        print(f"  Hash: {result.module_hash}")
        print(f"  C++ size: {len(result.cpp_source)} bytes")
        print(f"  Python size: {len(result.python_source)} bytes")
    else:
        print(f"✗ Direct extraction failed: {result.error_message}")


if __name__ == "__main__":
    main()
