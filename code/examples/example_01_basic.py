"""
Example 1: Basic Warp Kernel
Simple kernel to add two arrays element-wise.
"""

import warp as wp
import numpy as np

# Initialize Warp
wp.init()

# Define a simple kernel
@wp.kernel
def add_arrays(a: wp.array(dtype=float),
               b: wp.array(dtype=float),
               c: wp.array(dtype=float)):
    """Add two arrays element-wise."""
    i = wp.tid()  # thread id
    c[i] = a[i] + b[i]


def main():
    # Array size
    n = 10
    
    # Create input arrays
    a = wp.array(np.arange(n, dtype=np.float32), dtype=float)
    b = wp.array(np.arange(n, dtype=np.float32) * 2.0, dtype=float)
    c = wp.zeros(n, dtype=float)
    
    print(f"Input a: {a.numpy()}")
    print(f"Input b: {b.numpy()}")
    
    # Launch kernel
    wp.launch(kernel=add_arrays, dim=n, inputs=[a, b, c])
    
    # Synchronize and print results
    wp.synchronize()
    print(f"Output c: {c.numpy()}")
    
    # Verify
    expected = a.numpy() + b.numpy()
    if np.allclose(c.numpy(), expected):
        print("✓ Test passed!")
    else:
        print("✗ Test failed!")
    
    return True


if __name__ == "__main__":
    main()
