"""Explore IR extraction from warp kernels."""
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
    
    print("=== Module Attributes ===")
    for attr in sorted(dir(module)):
        if not attr.startswith('_'):
            val = getattr(module, attr)
            if isinstance(val, str) and len(val) < 200:
                print(f"  {attr}: '{val[:100]}...' " if len(val) > 100 else f"  {attr}: '{val}'")
            elif isinstance(val, str):
                print(f"  {attr}: str of len {len(val)}")
            elif not callable(val):
                print(f"  {attr}: {type(val).__name__} = {repr(val)[:80]}")
    
    print("\n=== Looking for generated code ===")
    
    # Check the builder
    if hasattr(module, 'builder') and module.builder:
        builder = module.builder
        print(f"Builder type: {type(builder)}")
        for attr in sorted(dir(builder)):
            if not attr.startswith('_'):
                val = getattr(builder, attr)
                if isinstance(val, str):
                    print(f"  {attr}: str len={len(val)}")
    
    # Check code gen options
    print("\n=== Checking for codegen output ===")
    
    # The generated C++ code is often in the module's code attribute
    if hasattr(module, 'code'):
        print(f"module.code: {len(module.code)} chars")
        print("First 500 chars:")
        print(module.code[:500])
    
    # Check cpu_module
    if hasattr(module, 'cpu_module') and module.cpu_module:
        cpu_mod = module.cpu_module
        print(f"\n=== CPU Module: {type(cpu_mod)} ===")
        for attr in sorted(dir(cpu_mod)):
            if not attr.startswith('_'):
                print(f"  {attr}")

if __name__ == "__main__":
    main()
