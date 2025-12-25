"""Dot product kernel test using atomic operations."""
import warp as wp

wp.init()

@wp.kernel
def dot_product(a: wp.array(dtype=float), b: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    wp.atomic_add(result, 0, a[tid] * b[tid])

if __name__ == "__main__":
    n = 10
    a = wp.array([float(i) for i in range(n)], dtype=float)  # 0, 1, 2, ..., 9
    b = wp.array([float(i) for i in range(n)], dtype=float)  # 0, 1, 2, ..., 9
    result = wp.zeros(1, dtype=float)
    
    wp.launch(dot_product, dim=n, inputs=[a, b, result])
    
    computed = result.numpy()[0]
    expected = sum(i*i for i in range(n))  # 0 + 1 + 4 + 9 + ... = 285
    print(f"Dot product result: {computed}")
    print(f"Expected: {expected}")
    print(f"Match: {abs(computed - expected) < 1e-6}")
