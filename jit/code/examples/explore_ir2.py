"""Explore IR extraction from warp kernels - Part 2."""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    """Add two arrays."""
    i = wp.tid()
    c[i] = a[i] + b[i]

def main():
    # Trigger compilation
    n = 4
    a = wp.array(np.ones(n, dtype=np.float32), device="cpu")
    b = wp.array(np.ones(n, dtype=np.float32) * 2, device="cpu")
    c = wp.zeros(n, dtype=float, device="cpu")
    wp.launch(simple_add, dim=n, inputs=[a, b, c], device="cpu")
    
    module = simple_add.module
    
    print("=== ModuleExec Exploration ===")
    for key, exec_obj in module.execs.items():
        print(f"\nExec key: {key}")
        print(f"Exec type: {type(exec_obj)}")
        for attr in sorted(dir(exec_obj)):
            if not attr.startswith('_'):
                val = getattr(exec_obj, attr)
                if isinstance(val, str):
                    if len(val) < 200:
                        print(f"  {attr}: '{val}'")
                    else:
                        print(f"  {attr}: str len={len(val)}")
                        print(f"    First 300 chars: {val[:300]}")
                elif not callable(val):
                    print(f"  {attr}: {type(val).__name__} = {repr(val)[:100]}")
    
    print("\n=== ModuleHasher Exploration ===")
    for key, hasher in module.hashers.items():
        print(f"\nHasher key: {key}")
        for attr in sorted(dir(hasher)):
            if not attr.startswith('_'):
                val = getattr(hasher, attr)
                if isinstance(val, str):
                    if len(val) < 200:
                        print(f"  {attr}: '{val}'")
                    else:
                        print(f"  {attr}: str len={len(val)}")
                        print(f"    First 500 chars:\n{val[:500]}")
                elif not callable(val):
                    print(f"  {attr}: {type(val).__name__}")

if __name__ == "__main__":
    main()
