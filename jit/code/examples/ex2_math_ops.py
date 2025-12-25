"""Example 2: Mathematical operations kernel."""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def math_ops(x: wp.array(dtype=float), out: wp.array(dtype=float)):
    """Apply sin, cos, exp operations."""
    i = wp.tid()
    val = x[i]
    # Compute: sin(x)^2 + cos(x)^2 should equal 1
    out[i] = wp.sin(val) * wp.sin(val) + wp.cos(val) * wp.cos(val)

def main():
    n = 5
    x = wp.array(np.linspace(0, np.pi, n, dtype=np.float32), device="cpu")
    out = wp.zeros(n, dtype=float, device="cpu")

    wp.launch(math_ops, dim=n, inputs=[x, out], device="cpu")
    
    print("Math Ops Example (sin^2 + cos^2 = 1):")
    print(f"  x = {x.numpy()}")
    print(f"  result = {out.numpy()}")
    print("  SUCCESS" if np.allclose(out.numpy(), np.ones(n)) else "  FAILED")

if __name__ == "__main__":
    main()
