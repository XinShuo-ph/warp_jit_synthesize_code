"""Explore warp kernel internals to find IR/code generation."""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

@wp.kernel
def multiply_scalar(arr: wp.array(dtype=float), scalar: float):
    tid = wp.tid()
    arr[tid] = arr[tid] * scalar

# Force compilation
n = 10
a = wp.array(np.ones(n, dtype=np.float32))
b = wp.array(np.ones(n, dtype=np.float32))
c = wp.zeros(n, dtype=float)
wp.launch(simple_add, dim=n, inputs=[a, b, c])
wp.synchronize()

print("=" * 60)
print("Exploring Kernel Object")
print("=" * 60)

print("\n1. Kernel attributes (non-private):")
for attr in sorted(dir(simple_add)):
    if not attr.startswith('_'):
        val = getattr(simple_add, attr, None)
        if not callable(val):
            print(f"  {attr}: {type(val).__name__} = {repr(val)[:80]}")

print("\n2. Kernel.module attributes:")
module = simple_add.module
for attr in sorted(dir(module)):
    if not attr.startswith('_'):
        val = getattr(module, attr, None)
        if not callable(val):
            print(f"  {attr}: {type(val).__name__}")

print("\n3. Kernel.adj (Adjoint) attributes:")
adj = simple_add.adj
for attr in sorted(dir(adj)):
    if not attr.startswith('_'):
        val = getattr(adj, attr, None)
        if not callable(val):
            print(f"  {attr}: {type(val).__name__}")

print("\n4. Source code (adj.source):")
print("-" * 40)
print(adj.source)
print("-" * 40)

print("\n5. Checking module for generated code:")
if hasattr(module, 'cpu_module'):
    print(f"  cpu_module: {module.cpu_module}")
if hasattr(module, 'cuda_module'):
    print(f"  cuda_module: {module.cuda_module}")
if hasattr(module, 'code'):
    print(f"  code length: {len(module.code) if module.code else 0}")
if hasattr(module, 'builder'):
    print(f"  builder: {module.builder}")

print("\n6. Looking at module.codegen attributes:")
if hasattr(module, 'builder') and module.builder:
    builder = module.builder
    print("  Builder attributes:")
    for attr in sorted(dir(builder)):
        if not attr.startswith('_'):
            val = getattr(builder, attr, None)
            if not callable(val):
                print(f"    {attr}: {type(val).__name__}")
