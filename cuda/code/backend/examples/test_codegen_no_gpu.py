import warp as wp
import warp._src.context
import numpy as np

wp.init()

@wp.kernel
def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def generate_cuda_code(kernel):
    module = kernel.module
    hasher = warp._src.context.ModuleHasher(module)
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", True)
    
    # We want to force CUDA generation
    builder = warp._src.context.ModuleBuilder(module, options, hasher)
    source = builder.codegen(device="cuda")
    return source

try:
    print("Attempting to generate CUDA code without GPU...")
    cuda_source = generate_cuda_code(simple_add)
    print("Success! Generated CUDA source length:", len(cuda_source))
    print("Sample lines:")
    print("\n".join(cuda_source.split("\n")[:20]))
    
    if "cuda_kernel" in cuda_source or "__global__" in cuda_source:
        print("Confirmed: Output contains CUDA keywords.")
    else:
        print("Warning: Output might not be CUDA code.")
        
except Exception as e:
    print(f"Failed: {e}")
