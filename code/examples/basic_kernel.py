#!/usr/bin/env python3
"""Basic warp kernel to verify installation and understand compilation."""

import warp as wp

# Initialize warp
wp.init()

@wp.kernel
def simple_add(a: wp.array(dtype=float),
                b: wp.array(dtype=float),
                c: wp.array(dtype=float)):
    """Simple kernel that adds two arrays element-wise."""
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def main():
    # Create test data
    n = 10
    a = wp.array([float(i) for i in range(n)], dtype=float)
    b = wp.array([float(i * 2) for i in range(n)], dtype=float)
    c = wp.zeros(n, dtype=float)
    
    print("Running simple_add kernel...")
    print(f"Input a: {a.numpy()}")
    print(f"Input b: {b.numpy()}")
    
    # Launch kernel
    wp.launch(kernel=simple_add, dim=n, inputs=[a, b, c])
    
    # Synchronize and get results
    wp.synchronize()
    result = c.numpy()
    print(f"Output c: {result}")
    
    # Verify correctness
    expected = [float(i + i*2) for i in range(n)]
    assert all(abs(result[i] - expected[i]) < 1e-6 for i in range(n)), "Results don't match!"
    print("✓ Test passed!")
    
    return result

if __name__ == "__main__":
    # Run twice to verify consistency
    print("=== Run 1 ===")
    result1 = main()
    print("\n=== Run 2 ===")
    result2 = main()
    
    # Verify both runs match
    assert all(abs(result1[i] - result2[i]) < 1e-6 for i in range(len(result1))), "Runs don't match!"
    print("\n✓ Both runs produced identical results")
