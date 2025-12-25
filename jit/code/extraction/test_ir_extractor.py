"""Test IR extraction with various kernel types."""
import warp as wp
import numpy as np
from ir_extractor import extract_ir, get_kernel_source, extract_pair

wp.init()

# ============================================================================
# Test Case 1: Basic Arithmetic
# ============================================================================
@wp.kernel
def kernel_arithmetic(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    """Basic arithmetic: c = a * 2 + b."""
    i = wp.tid()
    c[i] = a[i] * 2.0 + b[i]


# ============================================================================
# Test Case 2: Loop-based Kernel
# ============================================================================
@wp.kernel
def kernel_loop(input: wp.array(dtype=float), output: wp.array(dtype=float), n: int):
    """Sum elements using explicit loop."""
    i = wp.tid()
    total = float(0.0)
    for j in range(n):
        total = total + input[j]
    output[i] = total


# ============================================================================
# Test Case 3: Conditional Kernel
# ============================================================================
@wp.kernel
def kernel_conditional(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    """Apply ReLU-like operation: y = max(x, 0)."""
    i = wp.tid()
    val = x[i]
    if val > 0.0:
        y[i] = val
    else:
        y[i] = 0.0


# ============================================================================
# Test Case 4: Vector/Matrix Operations
# ============================================================================
@wp.kernel
def kernel_matrix(A: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    """Matrix-vector multiply."""
    i = wp.tid()
    out[i] = A[i] @ v[i]


# ============================================================================
# Test Case 5: Multiple Math Functions
# ============================================================================
@wp.kernel
def kernel_math_funcs(x: wp.array(dtype=float), y: wp.array(dtype=float)):
    """Apply various math functions."""
    i = wp.tid()
    val = x[i]
    # Chain of math operations
    y[i] = wp.sqrt(wp.abs(wp.sin(val) * wp.cos(val))) + wp.exp(-val * val)


# ============================================================================
# Test Runner
# ============================================================================
def test_kernel(kernel, name: str):
    """Test a single kernel and print results."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    
    # Get source
    source = get_kernel_source(kernel)
    print(f"Python Source ({len(source)} chars):")
    print(source[:300] + "..." if len(source) > 300 else source)
    
    # Get IR
    ir = extract_ir(kernel, "cpu")
    print(f"\nGenerated IR ({len(ir)} chars):")
    
    # Show key parts of IR
    if ir:
        # Find forward kernel
        if "_cpu_kernel_forward" in ir:
            start = ir.find("void " + kernel.adj.fun_name)
            if start == -1:
                start = ir.find("void ")
            end = ir.find("}", start) + 1
            print(ir[start:end][:500] + "..." if end - start > 500 else ir[start:end])
        print(f"\n✓ IR extracted successfully")
        return True
    else:
        print("✗ Failed to extract IR")
        return False


def run_all_tests():
    """Run all test cases."""
    print("="*60)
    print("IR Extraction Test Suite")
    print("="*60)
    
    tests = [
        (kernel_arithmetic, "Basic Arithmetic"),
        (kernel_loop, "Loop-based Kernel"),
        (kernel_conditional, "Conditional Kernel"),
        (kernel_matrix, "Matrix Operations"),
        (kernel_math_funcs, "Math Functions"),
    ]
    
    results = []
    for kernel, name in tests:
        try:
            success = test_kernel(kernel, name)
            results.append((name, success))
        except Exception as e:
            print(f"✗ Error: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for _, success in results if success)
    for name, success in results:
        print(f"  {'✓' if success else '✗'} {name}")
    print(f"\nTotal: {passed}/{len(results)} passed")
    
    return passed == len(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
