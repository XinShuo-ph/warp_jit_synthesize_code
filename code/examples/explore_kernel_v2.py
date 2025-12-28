"""Explore kernel IR extraction more deeply."""
import warp as wp
import os

wp.init()


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


if __name__ == "__main__":
    # Force compilation
    n = 10
    a = wp.zeros(n, dtype=float)
    b = wp.zeros(n, dtype=float)
    c = wp.zeros(n, dtype=float)
    wp.launch(add_kernel, dim=n, inputs=[a, b, c])
    wp.synchronize()
    
    mod = add_kernel.module
    
    print("=== Module execs ===")
    print("execs:", mod.execs)
    for key, value in mod.execs.items():
        print(f"  {key}: {type(value)}")
        if value:
            exec_attrs = [attr for attr in dir(value) if not attr.startswith('_')]
            print(f"    attributes: {exec_attrs}")
    
    # Check kernel cache
    print("\n=== Kernel cache ===")
    cache_dir = os.path.expanduser("~/.cache/warp/1.10.1")
    if os.path.exists(cache_dir):
        print(f"Cache dir: {cache_dir}")
        files = os.listdir(cache_dir)[:10]
        print(f"Files (first 10): {files}")
        
        # Find files for our module
        mod_id = mod.get_module_identifier()
        print(f"\nModule identifier: {mod_id}")
        
        # Look for source files
        for f in os.listdir(cache_dir):
            if mod.name.split('.')[-1] in f:
                print(f"  Found: {f}")
                if f.endswith('.cpp') or f.endswith('.cu'):
                    filepath = os.path.join(cache_dir, f)
                    print(f"  Reading {filepath}...")
                    with open(filepath, 'r') as fp:
                        content = fp.read()
                        print(content[:2000])
                        print("...")
    
    # Try building adjoint manually
    print("\n=== Adjoint build ===")
    adj = add_kernel.adj
    # Build adjoint to get generated code
    # Check the adj object after build
    if hasattr(adj, 'body_forward_lines'):
        print("body_forward_lines:", adj.body_forward_lines[:5])
    if hasattr(adj, 'body_reverse_lines'):
        print("body_reverse_lines:", adj.body_reverse_lines[:5])
