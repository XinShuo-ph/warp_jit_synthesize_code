"""Simple warp kernel example to understand basic compilation."""

import warp as wp

# Initialize warp
wp.init()

# Define a simple kernel
@wp.kernel
def simple_add(a: wp.array(dtype=float),
                b: wp.array(dtype=float),
                c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

# Create arrays
n = 10
a = wp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=float)
b = wp.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0], dtype=float)
c = wp.zeros(n, dtype=float)

# Launch kernel
wp.launch(simple_add, dim=n, inputs=[a, b, c])

# Synchronize and print results
wp.synchronize()
print("a:", a.numpy())
print("b:", b.numpy())
print("c:", c.numpy())
print("\nSimple kernel executed successfully!")
