"""
Test script to understand CPU vs CUDA code generation in Warp.
"""
import warp as wp
import warp._src.context as ctx

# Simple test kernel
@wp.kernel
def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]

def extract_code(kernel, device: str):
    """Extract generated code for a device."""
    module = kernel.module
    hasher = ctx.ModuleHasher(module)
    
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", False)
    options.setdefault("mode", "release")
    
    builder = ctx.ModuleBuilder(module, options, hasher)
    code = builder.codegen(device)
    
    return code

if __name__ == "__main__":
    wp.init()
    
    print("=" * 80)
    print("Testing CPU vs CUDA Code Generation")
    print("=" * 80)
    
    # Extract CPU code
    print("\n--- CPU Code (first 100 lines) ---")
    cpu_code = extract_code(test_kernel, "cpu")
    cpu_lines = cpu_code.split('\n')[:100]
    for i, line in enumerate(cpu_lines, 1):
        print(f"{i:3d}: {line}")
    
    print(f"\nTotal CPU code lines: {len(cpu_code.split(chr(10)))}")
    
    # Try to extract CUDA code (will fail without GPU but we can see the error)
    print("\n--- CUDA Code (first 100 lines) ---")
    try:
        cuda_code = extract_code(test_kernel, "cuda")
        cuda_lines = cuda_code.split('\n')[:100]
        for i, line in enumerate(cuda_lines, 1):
            print(f"{i:3d}: {line}")
        
        print(f"\nTotal CUDA code lines: {len(cuda_code.split(chr(10)))}")
        
        # Look for CUDA-specific patterns
        print("\n--- CUDA-specific patterns ---")
        if "__global__" in cuda_code:
            print("✓ Found __global__ (CUDA kernel qualifier)")
        if "threadIdx" in cuda_code:
            print("✓ Found threadIdx (CUDA built-in)")
        if "blockIdx" in cuda_code:
            print("✓ Found blockIdx (CUDA built-in)")
        if "__syncthreads" in cuda_code:
            print("✓ Found __syncthreads (CUDA synchronization)")
        
    except Exception as e:
        print(f"Error extracting CUDA code: {e}")
        print("(This is expected without a GPU)")
    
    # Save sample outputs
    with open("/workspace/cuda/notes/sample_cpu_code.txt", "w") as f:
        f.write(cpu_code)
    print(f"\nSaved full CPU code to: /workspace/cuda/notes/sample_cpu_code.txt")
    
    try:
        with open("/workspace/cuda/notes/sample_cuda_code.txt", "w") as f:
            f.write(cuda_code)
        print(f"Saved full CUDA code to: /workspace/cuda/notes/sample_cuda_code.txt")
    except:
        pass
