"""Extract generated C/C++ code from warp kernels."""
import warp as wp
import warp._src.codegen
import warp._src.context
import numpy as np

wp.init()

@wp.kernel
def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

@wp.kernel
def dot_product(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])

@wp.kernel
def matrix_vector_mult(m: wp.array(dtype=wp.mat33), v: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    out[tid] = m[tid] * v[tid]

# Force compilation by launching
n = 10
a = wp.array(np.ones(n, dtype=np.float32))
b = wp.array(np.ones(n, dtype=np.float32))  
c = wp.zeros(n, dtype=float)
wp.launch(simple_add, dim=n, inputs=[a, b, c])
wp.synchronize()


def extract_kernel_code(kernel, device="cpu"):
    """Extract generated C/C++ code for a kernel."""
    module = kernel.module
    
    # Get or create hasher for the module
    hasher = warp._src.context.ModuleHasher(module)
    
    # Create options dict
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", True)
    options.setdefault("mode", "release")
    
    # Create a builder to generate code
    builder = warp._src.context.ModuleBuilder(module, options, hasher)
    
    # Generate the code
    source = builder.codegen(device)
    
    return source


def extract_python_source(kernel):
    """Extract the original Python source for a kernel."""
    return kernel.adj.source


if __name__ == "__main__":
    print("=" * 70)
    print("Extracting Warp Kernel Code")
    print("=" * 70)
    
    kernels = [simple_add, dot_product, matrix_vector_mult]
    
    for kernel in kernels:
        print(f"\n{'=' * 70}")
        print(f"Kernel: {kernel.key}")
        print("=" * 70)
        
        print("\n--- Python Source ---")
        print(extract_python_source(kernel))
        
        print("\n--- Generated C++ Code (CPU) ---")
        cpp_code = extract_kernel_code(kernel, device="cpu")
        # Print just the first 200 lines to keep it manageable
        lines = cpp_code.split('\n')
        if len(lines) > 200:
            print('\n'.join(lines[:200]))
            print(f"\n... [{len(lines) - 200} more lines]")
        else:
            print(cpp_code)
        
        print("\n" + "=" * 70)
