"""Example 1: Basic warp kernel - vector addition."""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def vector_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    """Add two vectors element-wise."""
    i = wp.tid()
    c[i] = a[i] + b[i]

def main():
    n = 10
    a = wp.array(np.arange(n, dtype=np.float32), device="cpu")
    b = wp.array(np.arange(n, dtype=np.float32) * 2, device="cpu")
    c = wp.zeros(n, dtype=float, device="cpu")

    wp.launch(vector_add, dim=n, inputs=[a, b, c], device="cpu")
    
    print("Vector Add Example:")
    print(f"  a = {a.numpy()}")
    print(f"  b = {b.numpy()}")
    print(f"  c = a + b = {c.numpy()}")
    print("  SUCCESS" if np.allclose(c.numpy(), a.numpy() + b.numpy()) else "  FAILED")

if __name__ == "__main__":
    main()
