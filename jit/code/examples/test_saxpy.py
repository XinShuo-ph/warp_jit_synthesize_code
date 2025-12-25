"""SAXPY (Single-precision A*X Plus Y) kernel test."""
import warp as wp

wp.init()

@wp.kernel
def saxpy(a: float, x: wp.array(dtype=float), y: wp.array(dtype=float), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = a * x[tid] + y[tid]

if __name__ == "__main__":
    n = 8
    a = 2.0
    x = wp.array([float(i) for i in range(n)], dtype=float)
    y = wp.array([float(i * 10) for i in range(n)], dtype=float)
    out = wp.zeros(n, dtype=float)
    
    wp.launch(saxpy, dim=n, inputs=[a, x, y, out])
    
    result = out.numpy()
    expected = [a * i + i * 10 for i in range(n)]
    print(f"SAXPY result: {list(result)}")
    print(f"Expected: {expected}")
    print(f"Match: {all(abs(r - e) < 1e-6 for r, e in zip(result, expected))}")
