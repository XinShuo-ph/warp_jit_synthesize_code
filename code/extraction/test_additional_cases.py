#!/usr/bin/env python3
"""
Additional diverse test cases for IR extraction
Covers: structs, while loops, nested conditionals, math functions, etc.
"""

import warp as wp
from ir_extractor import IRExtractor, IRExtractorError
import json
import os

wp.init()

# Test 6: Custom struct
@wp.struct
class Particle:
    pos: wp.vec3
    vel: wp.vec3
    mass: float

@wp.kernel
def update_particles(particles: wp.array(dtype=Particle),
                     dt: float):
    tid = wp.tid()
    p = particles[tid]
    
    # Update position
    p.pos = p.pos + p.vel * dt
    
    # Apply gravity
    gravity = wp.vec3(0.0, -9.8, 0.0)
    p.vel = p.vel + gravity * dt
    
    particles[tid] = p

# Test 7: While loop
@wp.kernel
def binary_search(sorted_array: wp.array(dtype=float),
                  target: float,
                  result: wp.array(dtype=int)):
    tid = wp.tid()
    
    if tid == 0:  # Only first thread does the search
        n = sorted_array.shape[0]
        left = int(0)
        right = int(n - 1)
        found = int(-1)
        
        while left <= right:
            mid = (left + right) // 2
            mid_val = sorted_array[mid]
            
            if mid_val == target:
                found = mid
                break
            elif mid_val < target:
                left = mid + 1
            else:
                right = mid - 1
        
        result[0] = found

# Test 8: Nested conditionals and math
@wp.kernel
def classify_and_transform(x: wp.array(dtype=float),
                           y: wp.array(dtype=float)):
    tid = wp.tid()
    val = x[tid]
    
    result = float(0.0)
    
    if val < -1.0:
        # Use exponential for very negative
        result = wp.exp(val)
    elif val < 0.0:
        # Use square for slightly negative
        result = val * val
    elif val < 1.0:
        # Use identity for small positive
        result = val
    elif val < 10.0:
        # Use sqrt for medium positive
        result = wp.sqrt(val)
    else:
        # Use log for large positive
        result = wp.log(val)
    
    y[tid] = result

# Test 9: Matrix operations
@wp.kernel
def matrix_multiply_kernel(A: wp.array2d(dtype=float),
                           B: wp.array2d(dtype=float),
                           C: wp.array2d(dtype=float)):
    i, j = wp.tid()
    
    n = A.shape[1]  # columns of A
    sum_val = float(0.0)
    
    for k in range(n):
        sum_val = sum_val + A[i, k] * B[k, j]
    
    C[i, j] = sum_val

# Test 10: Trigonometric functions
@wp.kernel
def wave_simulation(x: wp.array(dtype=float),
                    t: float,
                    amplitude: float,
                    frequency: float,
                    phase: float,
                    output: wp.array(dtype=float)):
    tid = wp.tid()
    pos = x[tid]
    
    # Compute wave
    wave = amplitude * wp.sin(frequency * pos + phase + t)
    
    # Add some harmonics
    wave = wave + 0.5 * amplitude * wp.sin(2.0 * frequency * pos + phase)
    wave = wave + 0.25 * amplitude * wp.cos(3.0 * frequency * pos)
    
    output[tid] = wave

# Test 11: Bitwise operations
@wp.kernel
def bit_operations(a: wp.array(dtype=int),
                   b: wp.array(dtype=int),
                   result: wp.array(dtype=int)):
    tid = wp.tid()
    
    x = a[tid]
    y = b[tid]
    
    # Various bit ops
    and_result = x & y
    or_result = x | y
    xor_result = x ^ y
    
    # Combine them
    result[tid] = (and_result + or_result) ^ xor_result

# Test 12: Min/max and clamp
@wp.kernel
def bounds_and_clamp(values: wp.array(dtype=float),
                     lower: wp.array(dtype=float),
                     upper: wp.array(dtype=float),
                     output: wp.array(dtype=float)):
    tid = wp.tid()
    
    val = values[tid]
    lo = lower[tid]
    hi = upper[tid]
    
    # Use min/max
    val = wp.max(val, lo)
    val = wp.min(val, hi)
    
    # Equivalent to clamp
    output[tid] = wp.clamp(values[tid], lo, hi)

# Test 13: Atomic operations
@wp.kernel
def atomic_accumulate(input: wp.array(dtype=float),
                      output: wp.array(dtype=float),
                      bin_size: float):
    tid = wp.tid()
    val = input[tid]
    
    # Compute bin index
    bin_idx = int(val / bin_size)
    
    # Atomic add to histogram
    if bin_idx >= 0 and bin_idx < output.shape[0]:
        wp.atomic_add(output, bin_idx, 1.0)

# Test 14: Vector math
@wp.kernel
def vector_operations(a: wp.array(dtype=wp.vec3),
                      b: wp.array(dtype=wp.vec3),
                      dot_out: wp.array(dtype=float),
                      cross_out: wp.array(dtype=wp.vec3),
                      length_out: wp.array(dtype=float)):
    tid = wp.tid()
    
    v1 = a[tid]
    v2 = b[tid]
    
    # Dot product
    dot_out[tid] = wp.dot(v1, v2)
    
    # Cross product
    cross_out[tid] = wp.cross(v1, v2)
    
    # Length
    length_out[tid] = wp.length(v1)

# Test 15: Quaternion operations
@wp.kernel
def quaternion_rotate(quaternions: wp.array(dtype=wp.quat),
                      vectors: wp.array(dtype=wp.vec3),
                      output: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    
    q = quaternions[tid]
    v = vectors[tid]
    
    # Rotate vector by quaternion
    rotated = wp.quat_rotate(q, v)
    
    output[tid] = rotated

def extract_and_save_batch():
    """Extract all additional test cases."""
    print("Extracting Additional Test Cases")
    print("=" * 60)
    
    # Create output directory
    output_dir = "/workspace/data/samples"
    os.makedirs(output_dir, exist_ok=True)
    
    extractor = IRExtractor()
    extractor.set_verbose(True)
    
    # Prepare all test configurations
    n = 10
    
    # Test 6: Particles
    particles = wp.array([Particle() for _ in range(n)], dtype=Particle)
    
    # Test 7: Binary search
    sorted_arr = wp.array([float(i) for i in range(20)], dtype=float)
    search_result = wp.zeros(1, dtype=int)
    
    # Test 8: Classify
    x8 = wp.array([float(i - 5) for i in range(n)], dtype=float)
    y8 = wp.zeros(n, dtype=float)
    
    # Test 9: Matrix multiply
    m, k, p = 4, 3, 5
    A = wp.zeros((m, k), dtype=float)
    B = wp.zeros((k, p), dtype=float)
    C = wp.zeros((m, p), dtype=float)
    
    # Test 10: Wave
    x10 = wp.array([float(i) * 0.1 for i in range(n)], dtype=float)
    wave_out = wp.zeros(n, dtype=float)
    
    # Test 11: Bit ops
    a11 = wp.array([i for i in range(n)], dtype=int)
    b11 = wp.array([i * 2 for i in range(n)], dtype=int)
    r11 = wp.zeros(n, dtype=int)
    
    # Test 12: Bounds
    vals = wp.array([float(i) for i in range(n)], dtype=float)
    lower = wp.array([2.0] * n, dtype=float)
    upper = wp.array([7.0] * n, dtype=float)
    bounded = wp.zeros(n, dtype=float)
    
    # Test 13: Atomic
    input13 = wp.array([float(i % 5) for i in range(n)], dtype=float)
    hist = wp.zeros(5, dtype=float)
    
    # Test 14: Vector ops
    vecs_a = wp.array([[1.0, 0.0, 0.0]] * n, dtype=wp.vec3)
    vecs_b = wp.array([[0.0, 1.0, 0.0]] * n, dtype=wp.vec3)
    dot_r = wp.zeros(n, dtype=float)
    cross_r = wp.zeros(n, dtype=wp.vec3)
    len_r = wp.zeros(n, dtype=float)
    
    # Test 15: Quaternions
    quats = wp.array([wp.quat_identity()] * n, dtype=wp.quat)
    vecs15 = wp.array([[1.0, 0.0, 0.0]] * n, dtype=wp.vec3)
    rot_out = wp.zeros(n, dtype=wp.vec3)
    
    configs = [
        ("test_06_particles", update_particles, {'dim': n, 'inputs': [particles, 0.1]}),
        ("test_07_binary_search", binary_search, {'dim': 1, 'inputs': [sorted_arr, 5.0, search_result]}),
        ("test_08_classify", classify_and_transform, {'dim': n, 'inputs': [x8, y8]}),
        ("test_09_matmul", matrix_multiply_kernel, {'dim': (m, p), 'inputs': [A, B, C]}),
        ("test_10_wave", wave_simulation, {'dim': n, 'inputs': [x10, 0.0, 1.0, 2.0, 0.0, wave_out]}),
        ("test_11_bitops", bit_operations, {'dim': n, 'inputs': [a11, b11, r11]}),
        ("test_12_bounds", bounds_and_clamp, {'dim': n, 'inputs': [vals, lower, upper, bounded]}),
        ("test_13_atomic", atomic_accumulate, {'dim': n, 'inputs': [input13, hist, 1.0]}),
        ("test_14_vectors", vector_operations, {'dim': n, 'inputs': [vecs_a, vecs_b, dot_r, cross_r, len_r]}),
        ("test_15_quaternions", quaternion_rotate, {'dim': n, 'inputs': [quats, vecs15, rot_out]}),
    ]
    
    # Extract all
    for test_name, kernel, launch_args in configs:
        print(f"\n{test_name}: {kernel.key}")
        try:
            # Launch
            wp.launch(kernel=kernel, **launch_args)
            wp.synchronize()
            
            # Extract
            ir = extractor.extract_ir(kernel, trigger_compile=False)
            
            # Validate
            valid, msg = ir.validate()
            if not valid:
                print(f"  ✗ Validation failed: {msg}")
                continue
            
            # Save
            output = ir.to_dict()
            output['test_name'] = test_name
            
            filename = os.path.join(output_dir, f"{test_name}.json")
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"  ✓ Saved: {len(ir.python_source)} bytes Python → {len(ir.cpp_code)} bytes C++")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
    
    print("\n" + "=" * 60)
    print("Batch extraction complete!")

if __name__ == "__main__":
    extract_and_save_batch()
