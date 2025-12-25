"""Simple warp kernel test."""
import warp as wp

wp.init()

@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

if __name__ == "__main__":
    n = 10
    a = wp.array([float(i) for i in range(n)], dtype=float)
    b = wp.array([float(i) for i in range(n)], dtype=float)
    c = wp.zeros(n, dtype=float)

    wp.launch(add_kernel, dim=n, inputs=[a, b, c])
    print("Result:", c.numpy())
    print("Expected:", [float(i*2) for i in range(n)])
    print("Kernel compiled and executed successfully!")
