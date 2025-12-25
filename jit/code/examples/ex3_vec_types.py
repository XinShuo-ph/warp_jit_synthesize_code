"""Example 3: Vector types and operations."""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def dot_product(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    """Compute dot product of 3D vectors."""
    i = wp.tid()
    out[i] = wp.dot(a[i], b[i])

@wp.kernel
def cross_product(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    """Compute cross product of 3D vectors."""
    i = wp.tid()
    out[i] = wp.cross(a[i], b[i])

def main():
    n = 3
    # Create orthogonal unit vectors
    a_np = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    b_np = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32)
    
    a = wp.array(a_np, dtype=wp.vec3, device="cpu")
    b = wp.array(b_np, dtype=wp.vec3, device="cpu")
    dot_out = wp.zeros(n, dtype=float, device="cpu")
    cross_out = wp.zeros(n, dtype=wp.vec3, device="cpu")

    wp.launch(dot_product, dim=n, inputs=[a, b, dot_out], device="cpu")
    wp.launch(cross_product, dim=n, inputs=[a, b, cross_out], device="cpu")
    
    print("Vector Types Example:")
    print(f"  a = {a.numpy()}")
    print(f"  b = {b.numpy()}")
    print(f"  dot(a, b) = {dot_out.numpy()}")
    print(f"  cross(a, b) = {cross_out.numpy()}")
    
    # Orthogonal vectors have dot product = 0
    success = np.allclose(dot_out.numpy(), np.zeros(n))
    print("  SUCCESS" if success else "  FAILED")

if __name__ == "__main__":
    main()
