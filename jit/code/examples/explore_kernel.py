"""Explore kernel compilation and IR extraction."""
import warp as wp
import numpy as np

wp.init()

@wp.kernel
def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
    """Simple kernel for exploration."""
    i = wp.tid()
    b[i] = a[i] * 2.0

def main():
    print("=== Kernel Object Attributes ===")
    for attr in sorted(dir(test_kernel)):
        if not attr.startswith('_'):
            val = getattr(test_kernel, attr)
            if not callable(val):
                print(f"  {attr}: {type(val).__name__}")
            else:
                print(f"  {attr}(): method")
    
    print("\n=== Key Properties ===")
    print(f"  key: {test_kernel.key}")
    
    # Check adj (adjoint) which contains codegen info
    if hasattr(test_kernel, 'adj'):
        adj = test_kernel.adj
        print("\n=== Adjoint (adj) Attributes ===")
        for attr in sorted(dir(adj)):
            if not attr.startswith('_'):
                val = getattr(adj, attr)
                if isinstance(val, str) and len(val) < 100:
                    print(f"  {attr}: '{val}'")
                elif isinstance(val, str):
                    print(f"  {attr}: str of len {len(val)}")
                elif not callable(val):
                    print(f"  {attr}: {type(val).__name__}")
    
    # Trigger compilation by launching
    n = 4
    a = wp.array(np.ones(n, dtype=np.float32), device="cpu")
    b = wp.zeros(n, dtype=float, device="cpu")
    wp.launch(test_kernel, dim=n, inputs=[a, b], device="cpu")
    
    print("\n=== After Launch - Module ===")
    module = test_kernel.module
    print(f"  module.name: {module.name}")
    print(f"  module.hash_module: {module.hash_module}")
    
    # Check if we can access generated code
    if hasattr(module, 'cpu_module') and module.cpu_module:
        print("\n=== CPU Module ===")
        cpu_mod = module.cpu_module
        for attr in dir(cpu_mod):
            if not attr.startswith('_'):
                print(f"  {attr}")

if __name__ == "__main__":
    main()
