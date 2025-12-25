import warp as wp
import numpy as np

@wp.kernel
def add_kernel(x: wp.array(dtype=float), y: wp.array(dtype=float), z: wp.array(dtype=float), n: int):
    tid = wp.tid()
    if tid < n:
        z[tid] = x[tid] + y[tid]

def run_example():
    wp.init()
    n = 100
    
    # Create input arrays
    x_np = np.arange(n, dtype=float)
    y_np = np.ones(n, dtype=float) * 10.0
    
    x = wp.from_numpy(x_np, dtype=float)
    y = wp.from_numpy(y_np, dtype=float)
    z = wp.zeros(n, dtype=float)
    
    # Launch kernel
    wp.launch(kernel=add_kernel, dim=n, inputs=[x, y, z, n])
    
    # Verify
    z_np = z.numpy()
    expected = x_np + y_np
    
    if np.allclose(z_np, expected):
        print("Vector Add: SUCCESS")
        print(f"First 5 elements: {z_np[:5]}")
    else:
        print("Vector Add: FAILED")
        print(f"Expected: {expected[:5]}")
        print(f"Got: {z_np[:5]}")

if __name__ == "__main__":
    run_example()
