"""
Analyze CPU vs CUDA code generation differences in Warp.

This script examines how Warp generates different code for CPU and CUDA backends.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "synthesis"))

import warp as wp
import warp._src.context as ctx

# Initialize warp
wp.init()


def generate_ir_for_device(kernel, device: str) -> str:
    """Generate IR code for a specific device."""
    module = kernel.module
    hasher = ctx.ModuleHasher(module)
    
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", False)
    options.setdefault("mode", "release")
    
    builder = ctx.ModuleBuilder(module, options, hasher)
    return builder.codegen(device)


def extract_forward_function(code: str, mangled_name: str, device: str) -> str:
    """Extract the forward kernel function from generated code."""
    import re
    forward_func_name = f"{mangled_name}_{device}_kernel_forward"
    pattern = rf'void {re.escape(forward_func_name)}\s*\([^)]*\)\s*\{{'
    
    match = re.search(pattern, code)
    if not match:
        return f"Function {forward_func_name} not found"
    
    start = match.start()
    brace_count = 0
    in_function = False
    end = start
    
    for i, char in enumerate(code[start:], start):
        if char == '{':
            brace_count += 1
            in_function = True
        elif char == '}':
            brace_count -= 1
            if in_function and brace_count == 0:
                end = i + 1
                break
    
    return code[start:end]


# Define test kernels - must be at module level for Warp
@wp.kernel
def simple_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
    """Simple element-wise addition."""
    tid = wp.tid()
    c[tid] = a[tid] + b[tid]


@wp.kernel
def atomic_sum(values: wp.array(dtype=float), result: wp.array(dtype=float)):
    """Atomic sum reduction."""
    tid = wp.tid()
    wp.atomic_add(result, 0, values[tid])


@wp.kernel
def vec_ops(a: wp.array(dtype=wp.vec3), b: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    """Vector dot product."""
    tid = wp.tid()
    out[tid] = wp.dot(a[tid], b[tid])


def analyze_kernel(kernel, name: str):
    """Analyze a kernel for both CPU and CUDA targets."""
    print(f"\n{'='*60}")
    print(f"Kernel: {name}")
    print('='*60)
    
    # Force module to be loaded/hashed
    module = kernel.module
    hasher = ctx.ModuleHasher(module)
    
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", False)
    options.setdefault("mode", "release")
    
    # This builds the module and sets up hashes
    builder = ctx.ModuleBuilder(module, options, hasher)
    
    mangled_name = kernel.get_mangled_name()
    
    # Generate CPU code
    cpu_code = builder.codegen("cpu")
    cpu_forward = extract_forward_function(cpu_code, mangled_name, "cpu")
    
    # Generate CUDA code (using same builder)
    cuda_code = builder.codegen("cuda")
    cuda_forward = extract_forward_function(cuda_code, mangled_name, "cuda")
    
    print("\n--- CPU Forward ---")
    print(cpu_forward)
    
    print("\n--- CUDA Forward ---")
    print(cuda_forward)
    
    # Key differences
    print("\n--- Key Differences ---")
    if "_cpu_kernel_forward" in cpu_forward and "_cuda_kernel_forward" in cuda_forward:
        print("- Function name suffix: _cpu_kernel_forward vs _cuda_kernel_forward")
    
    if "WP_CPU" in cpu_code:
        print("- CPU defines: WP_CPU macro")
    if "WP_CUDA" in cuda_code:
        print("- CUDA defines: WP_CUDA macro")
        
    return {
        "cpu_forward": cpu_forward,
        "cuda_forward": cuda_forward,
        "cpu_full": cpu_code,
        "cuda_full": cuda_code,
    }


def find_header_differences(cpu_code: str, cuda_code: str):
    """Compare headers and includes between CPU and CUDA code."""
    print("\n" + "="*60)
    print("Header Analysis")
    print("="*60)
    
    # Get first 100 lines of each
    cpu_lines = cpu_code.split('\n')[:100]
    cuda_lines = cuda_code.split('\n')[:100]
    
    # Find key patterns
    cpu_includes = [l for l in cpu_lines if '#include' in l]
    cuda_includes = [l for l in cuda_lines if '#include' in l]
    
    print("\nCPU includes:")
    for inc in cpu_includes[:10]:
        print(f"  {inc}")
    
    print("\nCUDA includes:")
    for inc in cuda_includes[:10]:
        print(f"  {inc}")
    
    # Find CUDA-specific constructs
    cuda_specific = []
    if '__global__' in cuda_code:
        cuda_specific.append("__global__ (CUDA kernel qualifier)")
    if '__device__' in cuda_code:
        cuda_specific.append("__device__ (CUDA device function)")
    if '__shared__' in cuda_code:
        cuda_specific.append("__shared__ (CUDA shared memory)")
    if 'blockIdx' in cuda_code:
        cuda_specific.append("blockIdx (CUDA block index)")
    if 'threadIdx' in cuda_code:
        cuda_specific.append("threadIdx (CUDA thread index)")
    if 'blockDim' in cuda_code:
        cuda_specific.append("blockDim (CUDA block dimension)")
        
    if cuda_specific:
        print("\nCUDA-specific constructs found:")
        for item in cuda_specific:
            print(f"  - {item}")


if __name__ == "__main__":
    print("Warp CPU vs CUDA Code Generation Analysis")
    print("="*60)
    print(f"Warp version: {wp.__version__}")
    print(f"CUDA available: {wp.is_cuda_available()}")
    print(f"CPU available: {wp.is_cpu_available()}")
    
    # Analyze each kernel
    results = {}
    results['simple_add'] = analyze_kernel(simple_add, 'simple_add')
    results['atomic_sum'] = analyze_kernel(atomic_sum, 'atomic_sum')
    results['vec_ops'] = analyze_kernel(vec_ops, 'vec_ops')
    
    # Analyze headers
    find_header_differences(results['simple_add']['cpu_full'], 
                           results['simple_add']['cuda_full'])
    
    print("\n" + "="*60)
    print("Summary: Key CPU vs CUDA Differences")
    print("="*60)
    print("""
1. Function naming: *_cpu_kernel_forward vs *_cuda_kernel_forward
2. File extension: .cpp (CPU) vs .cu (CUDA)
3. CUDA uses __global__, __device__, __shared__ qualifiers
4. CUDA uses blockIdx, threadIdx, blockDim for thread indexing
5. Both use same wp:: namespace and internal functions
6. IR extraction works identically - just pass device="cuda"
""")
