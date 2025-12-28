"""
Example 1: Simple kernel - basic array operations
"""
import warp as wp
import numpy as np

# Initialize warp
wp.init()

# Define a simple kernel
@wp.kernel
def simple_add(a: wp.array(dtype=float), 
                b: wp.array(dtype=float),
                c: wp.array(dtype=float)):
    """Add two arrays element-wise"""
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def run_example():
    # Create test data
    n = 10
    a_np = np.arange(n, dtype=np.float32)
    b_np = np.ones(n, dtype=np.float32) * 2.0
    
    # Transfer to warp arrays
    a = wp.array(a_np, dtype=wp.float32)
    b = wp.array(b_np, dtype=wp.float32)
    c = wp.zeros(n, dtype=wp.float32)
    
    # Launch kernel
    wp.launch(simple_add, dim=n, inputs=[a, b, c])
    
    # Get results
    result = c.numpy()
    
    print("Example 1: Simple Add Kernel")
    print(f"Input a: {a_np}")
    print(f"Input b: {b_np}")
    print(f"Output c: {result}")
    print(f"Expected: {a_np + b_np}")
    print(f"Match: {np.allclose(result, a_np + b_np)}")
    print()
    
    return np.allclose(result, a_np + b_np)

if __name__ == "__main__":
    success = run_example()
    print(f"Test {'PASSED' if success else 'FAILED'}")
