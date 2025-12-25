"""Explore warp kernel compilation to understand IR generation."""

import warp as wp
import os

# Enable verbose output
wp.config.verbose = True

# Initialize warp
wp.init()

print(f"Warp config: {dir(wp.config)}")

# Define a test kernel
@wp.kernel
def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0 + 1.0

# Get kernel info before compilation
print(f"\nKernel function: {test_kernel.func.__name__}")
print(f"Kernel module: {test_kernel.module.name}")
print(f"Kernel key: {test_kernel.key}")

# Access module and kernel internals
module = test_kernel.module
print(f"\nModule name: {module.name}")

# Create test data and launch to trigger compilation
n = 10
a = wp.array([float(i) for i in range(n)], dtype=float)
b = wp.zeros(n, dtype=float)

# Launch the kernel (this triggers compilation)
print("\n--- Launching kernel (will trigger compilation) ---")
wp.launch(test_kernel, dim=n, inputs=[a, b])
wp.synchronize()

print("\n--- After compilation ---")
print(f"Result: {b.numpy()}")

# Check if we can access the generated code
if hasattr(module, 'cpp_source'):
    print(f"\nC++ source available: {len(module.cpp_source)} characters")
    print("First 500 chars:")
    print(module.cpp_source[:500])

# Check cache directory
cache_dir = wp.config.kernel_cache_dir
print(f"\nChecking cache directory: {cache_dir}")
if cache_dir and os.path.exists(cache_dir):
    for root, dirs, files in os.walk(cache_dir):
        for f in files:
            filepath = os.path.join(root, f)
            print(f"  {filepath}")
            if f.endswith('.cpp') or f.endswith('.cu') or f.endswith('.ptx'):
                print(f"    -> Found potential IR/source file!")
