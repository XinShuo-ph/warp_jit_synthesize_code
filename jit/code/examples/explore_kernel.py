"""Explore kernel attributes to understand IR extraction."""
import warp as wp

wp.init()


@wp.kernel
def add_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


if __name__ == "__main__":
    # Explore kernel object
    print("=== Kernel attributes ===")
    kernel_attrs = [attr for attr in dir(add_kernel) if not attr.startswith('_')]
    print(kernel_attrs)
    
    print("\n=== Adjoint attributes ===")
    adj_attrs = [attr for attr in dir(add_kernel.adj) if not attr.startswith('_')]
    print(adj_attrs)
    
    # Build the kernel to get generated code
    print("\n=== Source code (Python) ===")
    print(add_kernel.adj.source)
    
    # Force compilation by launching
    n = 10
    a = wp.zeros(n, dtype=float)
    b = wp.zeros(n, dtype=float)
    c = wp.zeros(n, dtype=float)
    wp.launch(add_kernel, dim=n, inputs=[a, b, c])
    wp.synchronize()
    
    # Check module for generated code
    print("\n=== Module attributes ===")
    mod = add_kernel.module
    mod_attrs = [attr for attr in dir(mod) if not attr.startswith('_')]
    print(mod_attrs)
    
    # Try to get generated C++ source
    print("\n=== Checking for generated code ===")
    if hasattr(mod, 'cpu_module'):
        print("cpu_module:", mod.cpu_module)
    if hasattr(mod, 'cuda_module'):
        print("cuda_module:", mod.cuda_module)
    
    # Try adjoint's forward/reverse 
    print("\n=== Adjoint forward body ===")
    if hasattr(add_kernel.adj, 'body_forward'):
        print("body_forward:", add_kernel.adj.body_forward)
    if hasattr(add_kernel.adj, 'forward'):
        print("forward:", type(add_kernel.adj.forward))
    if hasattr(add_kernel.adj, 'body'):
        print("body:", add_kernel.adj.body)
