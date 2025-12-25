"""Test IR extraction with 5+ varied kernels."""
import warp as wp
from ir_extractor import extract_ir, extract_kernel_functions, IRPair

wp.init()


# Test 1: Basic arithmetic
@wp.kernel
def kernel_arithmetic(a: wp.array(dtype=float), b: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a[tid] * 2.0 + b[tid] - 1.0


# Test 2: Control flow (if statement)
@wp.kernel
def kernel_conditional(x: wp.array(dtype=float), out: wp.array(dtype=float), threshold: float):
    tid = wp.tid()
    if x[tid] > threshold:
        out[tid] = x[tid] * 2.0
    else:
        out[tid] = x[tid] * 0.5


# Test 3: Loop (for)
@wp.kernel
def kernel_loop(arr: wp.array(dtype=float), out: wp.array(dtype=float), iterations: int):
    tid = wp.tid()
    total = float(0.0)
    for i in range(iterations):
        total = total + arr[tid] * float(i)
    out[tid] = total


# Test 4: Vector operations
@wp.kernel
def kernel_vector(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), dt: float):
    tid = wp.tid()
    pos[tid] = pos[tid] + vel[tid] * dt


# Test 5: Atomics
@wp.kernel
def kernel_atomic(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])


# Test 6: User-defined function
@wp.func
def helper_square(x: float) -> float:
    return x * x


@wp.kernel
def kernel_with_func(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = helper_square(x[tid])


def test_kernel(kernel, name: str) -> bool:
    """Test IR extraction for a kernel."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print('='*60)
    
    try:
        ir_pair = extract_ir(kernel)
        
        print("\n--- Python Source ---")
        print(ir_pair.python_source.strip())
        
        print("\n--- C++ Forward (excerpt) ---")
        funcs = extract_kernel_functions(ir_pair.cpp_ir, kernel.key)
        if 'forward' in funcs:
            # Print first 800 chars or until end
            fwd = funcs['forward']
            print(fwd[:800] + ("..." if len(fwd) > 800 else ""))
        else:
            print("WARNING: Forward function not found!")
            return False
        
        print("\n✓ PASSED")
        return True
        
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def run_all_tests():
    """Run all IR extraction tests."""
    tests = [
        (kernel_arithmetic, "Basic arithmetic"),
        (kernel_conditional, "Control flow (if)"),
        (kernel_loop, "Loop (for)"),
        (kernel_vector, "Vector operations"),
        (kernel_atomic, "Atomics"),
        (kernel_with_func, "User-defined function"),
    ]
    
    results = []
    for kernel, name in tests:
        results.append(test_kernel(kernel, name))
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return True
    else:
        print("\n✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
