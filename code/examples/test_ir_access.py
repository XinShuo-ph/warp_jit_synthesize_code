"""Test programmatic access to generated IR."""

import warp as wp
import os

wp.init()

# Create a test kernel
@wp.kernel
def simple_mul(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] * b[tid]

# Launch to trigger compilation
n = 5
a = wp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
b = wp.array([2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
c = wp.zeros(n, dtype=float)

wp.launch(simple_mul, dim=n, inputs=[a, b, c])
wp.synchronize()

print("Python kernel code:")
print("=" * 60)
import inspect
print(inspect.getsource(simple_mul.func))
print("=" * 60)

# Access module internals
module = simple_mul.module
print(f"\nModule: {module.name}")
print(f"Module attributes: {[a for a in dir(module) if not a.startswith('_')]}")

# Try to access generated source
if hasattr(module, 'cpp_source'):
    print("\nGenerated C++ source (first 500 chars):")
    print(module.cpp_source[:500])

# Find the cache file by looking at what was created
cache_dir = wp.config.kernel_cache_dir
if cache_dir and os.path.exists(cache_dir):
    # Find the most recent module directory
    module_dirs = [d for d in os.listdir(cache_dir) if d.startswith('wp___main__')]
    if module_dirs:
        # Get the most recently modified one
        latest_dir = max([os.path.join(cache_dir, d) for d in module_dirs], key=os.path.getmtime)
        cpp_files = [f for f in os.listdir(latest_dir) if f.endswith('.cpp')]
        
        if cpp_files:
            cpp_file = os.path.join(latest_dir, cpp_files[0])
            print(f"\nFound generated C++ file: {cpp_file}")
            
            with open(cpp_file, 'r') as f:
                content = f.read()
                print("\nGenerated C++ IR (first 1000 chars):")
                print(content[:1000])
                print(f"\n... (total {len(content)} characters)")
