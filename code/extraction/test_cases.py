"""
Test cases for IR extraction - 5+ diverse kernels
"""

import warp as wp
import numpy as np
from pathlib import Path
import sys
import os

# Add extraction module to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from extraction.ir_extractor import IRExtractor

wp.init()

# Test Case 1: Simple arithmetic operations
@wp.kernel
def kernel_arithmetic(
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    c: wp.array(dtype=float)
):
    """Simple arithmetic: c = (a + b) * 2.0 - 1.0"""
    tid = wp.tid()
    c[tid] = (a[tid] + b[tid]) * 2.0 - 1.0


# Test Case 2: Vector operations
@wp.kernel
def kernel_vectors(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    forces: wp.array(dtype=wp.vec3),
    dt: float
):
    """Vector math: forces from positions and velocities"""
    tid = wp.tid()
    pos = positions[tid]
    vel = velocities[tid]
    
    # Compute distance from origin
    dist = wp.length(pos)
    
    # Spring force toward origin
    spring_force = wp.normalize(pos) * (-10.0 * dist)
    
    # Damping force
    damping = vel * (-0.5)
    
    forces[tid] = spring_force + damping


# Test Case 3: Control flow with conditionals
@wp.kernel
def kernel_conditionals(
    values: wp.array(dtype=float),
    categories: wp.array(dtype=int),
    results: wp.array(dtype=float)
):
    """Control flow: categorize and process values"""
    tid = wp.tid()
    val = values[tid]
    
    if val < -1.0:
        categories[tid] = 0
        results[tid] = val * val
    elif val < 1.0:
        categories[tid] = 1
        results[tid] = val
    else:
        categories[tid] = 2
        results[tid] = wp.sqrt(val)


# Test Case 4: Loops and accumulation
@wp.kernel
def kernel_loop(
    matrix: wp.array2d(dtype=float),
    row_sums: wp.array(dtype=float)
):
    """Loop: compute sum of each row"""
    tid = wp.tid()
    
    total = float(0.0)  # Must use float() for mutable loop variables
    for j in range(matrix.shape[1]):
        total = total + matrix[tid, j]
    
    row_sums[tid] = total


# Test Case 5: Atomic operations for parallel reduction
@wp.kernel
def kernel_atomic(
    values: wp.array(dtype=float),
    threshold: float,
    result: wp.array(dtype=float)
):
    """Atomic operations: count and sum values above threshold"""
    tid = wp.tid()
    val = values[tid]
    
    if val > threshold:
        wp.atomic_add(result, 0, val)
        wp.atomic_add(result, 1, 1.0)


# Test Case 6: Matrix operations
@wp.kernel
def kernel_matrix(
    A: wp.array2d(dtype=float),
    x: wp.array(dtype=float),
    y: wp.array(dtype=float)
):
    """Matrix-vector multiplication: y = A * x"""
    tid = wp.tid()
    
    sum_val = float(0.0)  # Must use float() for mutable loop variables
    for j in range(A.shape[1]):
        sum_val = sum_val + A[tid, j] * x[j]
    
    y[tid] = sum_val


# Test Case 7: Trigonometry and math functions
@wp.kernel
def kernel_math(
    x: wp.array(dtype=float),
    sin_x: wp.array(dtype=float),
    cos_x: wp.array(dtype=float),
    exp_x: wp.array(dtype=float)
):
    """Math functions: compute sin, cos, exp"""
    tid = wp.tid()
    val = x[tid]
    
    sin_x[tid] = wp.sin(val)
    cos_x[tid] = wp.cos(val)
    exp_x[tid] = wp.exp(val * 0.1)  # Scale to avoid overflow


def run_and_extract_all():
    """Run all test kernels and extract their IR"""
    
    extractor = IRExtractor()
    results = []
    
    # Create output directory
    output_dir = Path(__file__).parent.parent.parent / "data" / "test_cases"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("IR EXTRACTION TEST SUITE")
    print("=" * 80)
    
    # Test 1: Arithmetic
    print("\n1. Testing arithmetic kernel...")
    n = 10
    a = wp.array(np.random.randn(n).astype(np.float32))
    b = wp.array(np.random.randn(n).astype(np.float32))
    c = wp.zeros(n, dtype=wp.float32)
    wp.launch(kernel_arithmetic, dim=n, inputs=[a, b, c])
    
    ir_data = extractor.extract_ir(kernel_arithmetic)
    extractor.save_pair(
        ir_data['python_source'],
        ir_data['forward_function'],
        str(output_dir / "01_arithmetic.json")
    )
    results.append(("arithmetic", ir_data))
    print("   ✓ Extracted")
    
    # Test 2: Vectors
    print("\n2. Testing vector operations kernel...")
    n = 10
    positions = wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3)
    velocities = wp.array(np.random.randn(n, 3).astype(np.float32), dtype=wp.vec3)
    forces = wp.zeros(n, dtype=wp.vec3)
    wp.launch(kernel_vectors, dim=n, inputs=[positions, velocities, forces, 0.01])
    
    ir_data = extractor.extract_ir(kernel_vectors)
    extractor.save_pair(
        ir_data['python_source'],
        ir_data['forward_function'],
        str(output_dir / "02_vectors.json")
    )
    results.append(("vectors", ir_data))
    print("   ✓ Extracted")
    
    # Test 3: Conditionals
    print("\n3. Testing conditionals kernel...")
    n = 10
    values = wp.array(np.linspace(-2, 2, n).astype(np.float32))
    categories = wp.zeros(n, dtype=wp.int32)
    results_array = wp.zeros(n, dtype=wp.float32)
    wp.launch(kernel_conditionals, dim=n, inputs=[values, categories, results_array])
    
    ir_data = extractor.extract_ir(kernel_conditionals)
    extractor.save_pair(
        ir_data['python_source'],
        ir_data['forward_function'],
        str(output_dir / "03_conditionals.json")
    )
    results.append(("conditionals", ir_data))
    print("   ✓ Extracted")
    
    # Test 4: Loop
    print("\n4. Testing loop kernel...")
    rows, cols = 5, 8
    matrix = wp.array(np.random.randn(rows, cols).astype(np.float32), dtype=wp.float32)
    row_sums = wp.zeros(rows, dtype=wp.float32)
    wp.launch(kernel_loop, dim=rows, inputs=[matrix, row_sums])
    
    ir_data = extractor.extract_ir(kernel_loop)
    extractor.save_pair(
        ir_data['python_source'],
        ir_data['forward_function'],
        str(output_dir / "04_loop.json")
    )
    results.append(("loop", ir_data))
    print("   ✓ Extracted")
    
    # Test 5: Atomic
    print("\n5. Testing atomic operations kernel...")
    n = 20
    values = wp.array(np.random.randn(n).astype(np.float32))
    result = wp.zeros(2, dtype=wp.float32)
    wp.launch(kernel_atomic, dim=n, inputs=[values, 0.5, result])
    
    ir_data = extractor.extract_ir(kernel_atomic)
    extractor.save_pair(
        ir_data['python_source'],
        ir_data['forward_function'],
        str(output_dir / "05_atomic.json")
    )
    results.append(("atomic", ir_data))
    print("   ✓ Extracted")
    
    # Test 6: Matrix
    print("\n6. Testing matrix operations kernel...")
    rows, cols = 5, 8
    A = wp.array(np.random.randn(rows, cols).astype(np.float32), dtype=wp.float32)
    x = wp.array(np.random.randn(cols).astype(np.float32), dtype=wp.float32)
    y = wp.zeros(rows, dtype=wp.float32)
    wp.launch(kernel_matrix, dim=rows, inputs=[A, x, y])
    
    ir_data = extractor.extract_ir(kernel_matrix)
    extractor.save_pair(
        ir_data['python_source'],
        ir_data['forward_function'],
        str(output_dir / "06_matrix.json")
    )
    results.append(("matrix", ir_data))
    print("   ✓ Extracted")
    
    # Test 7: Math
    print("\n7. Testing math functions kernel...")
    n = 10
    x = wp.array(np.linspace(-3.14, 3.14, n).astype(np.float32))
    sin_x = wp.zeros(n, dtype=wp.float32)
    cos_x = wp.zeros(n, dtype=wp.float32)
    exp_x = wp.zeros(n, dtype=wp.float32)
    wp.launch(kernel_math, dim=n, inputs=[x, sin_x, cos_x, exp_x])
    
    ir_data = extractor.extract_ir(kernel_math)
    extractor.save_pair(
        ir_data['python_source'],
        ir_data['forward_function'],
        str(output_dir / "07_math.json")
    )
    results.append(("math", ir_data))
    print("   ✓ Extracted")
    
    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"\nTotal kernels extracted: {len(results)}")
    print(f"Output directory: {output_dir}")
    print("\nKernel details:")
    for name, ir_data in results:
        py_lines = len(ir_data['python_source'].split('\n'))
        ir_lines = len(ir_data['forward_function'].split('\n')) if ir_data['forward_function'] else 0
        print(f"  {name:15s}: {py_lines:3d} Python lines → {ir_lines:3d} IR lines")
    
    return results


if __name__ == "__main__":
    results = run_and_extract_all()
    print("\n✓ All test cases completed successfully!")
