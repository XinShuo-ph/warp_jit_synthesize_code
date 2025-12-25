"""
Explore warp kernel compilation mechanism and IR generation
"""
import warp as wp
import inspect

wp.init()

# Define a simple test kernel
@wp.kernel
def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    tid = wp.tid()
    b[tid] = a[tid] * 2.0 + 1.0

# Explore the kernel object
print("=" * 80)
print("KERNEL OBJECT EXPLORATION")
print("=" * 80)

print(f"\n1. Kernel type: {type(test_kernel)}")
print(f"   Kernel class: {test_kernel.__class__.__name__}")

# Get kernel attributes
print("\n2. Kernel attributes:")
for attr in dir(test_kernel):
    if not attr.startswith('_'):
        try:
            value = getattr(test_kernel, attr)
            if not callable(value):
                print(f"   {attr}: {type(value).__name__}")
        except:
            pass

# Check the module
print(f"\n3. Kernel module: {test_kernel.module}")
print(f"   Module type: {type(test_kernel.module)}")

# Explore the module
print("\n4. Module attributes:")
for attr in dir(test_kernel.module):
    if not attr.startswith('_'):
        try:
            value = getattr(test_kernel.module, attr)
            if not callable(value):
                print(f"   {attr}: {type(value).__name__} = {value if not isinstance(value, (dict, list)) or len(str(value)) < 50 else '...'}")
        except Exception as e:
            print(f"   {attr}: Error - {e}")

# Try to access kernel code
print("\n5. Looking for generated code:")
if hasattr(test_kernel, 'adj'):
    print(f"   Adjoint object: {test_kernel.adj}")
if hasattr(test_kernel.module, 'cpu_module'):
    print(f"   CPU module: {test_kernel.module.cpu_module}")
if hasattr(test_kernel.module, 'cuda_module'):
    print(f"   CUDA module: {test_kernel.module.cuda_module}")

# Check if there's a code generator
print("\n6. Checking for code generation:")
if hasattr(test_kernel.module, 'load'):
    print(f"   Module.load method exists")
if hasattr(test_kernel.module, 'unload'):
    print(f"   Module.unload method exists")

# Actually compile the kernel by launching it
print("\n7. Forcing kernel compilation...")
import numpy as np
n = 4
a = wp.array(np.ones(n, dtype=np.float32))
b = wp.zeros(n, dtype=wp.float32)
wp.launch(test_kernel, dim=n, inputs=[a, b])
print("   Kernel launched successfully")

# Check for generated code after compilation
print("\n8. Post-compilation exploration:")
if hasattr(test_kernel.module, 'cpu_module'):
    cpu_mod = test_kernel.module.cpu_module
    print(f"   CPU module: {cpu_mod}")
    if cpu_mod:
        print(f"   CPU module type: {type(cpu_mod)}")

# Check for source code
if hasattr(test_kernel.module, 'source'):
    print(f"\n9. Module source code available: {len(test_kernel.module.source)} chars")
    print("   First 500 chars:")
    print("   " + "-" * 76)
    print("   " + test_kernel.module.source[:500])

# Check kernel cache location
print(f"\n10. Kernel cache location:")
print(f"    {wp.config.kernel_cache_dir}")

# List cached files
import os
if os.path.exists(wp.config.kernel_cache_dir):
    print(f"\n11. Cache directory contents:")
    for root, dirs, files in os.walk(wp.config.kernel_cache_dir):
        level = root.replace(wp.config.kernel_cache_dir, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 4 * (level + 1)
        for file in files[:10]:  # Limit to first 10 files per directory
            print(f'{subindent}{file}')
        if len(files) > 10:
            print(f'{subindent}... and {len(files) - 10} more files')

print("\n" + "=" * 80)
