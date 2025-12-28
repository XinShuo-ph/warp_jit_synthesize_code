"""
CUDA Code Templates for standalone CUDA C++ generation.
These templates produce compilable .cu files without Warp runtime dependencies.
"""


def cuda_kernel_template(kernel_name, params, body, block_size=256):
    """
    Generate standalone CUDA kernel code.
    
    Args:
        kernel_name: Name of the kernel
        params: List of (name, type) tuples
        body: Kernel body code
        block_size: CUDA block size
    """
    param_decls = ", ".join([f"{ptype}* {pname}" for pname, ptype in params])
    
    return f'''
// CUDA Kernel: {kernel_name}
__global__ void {kernel_name}_kernel({param_decls}, int n)
{{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {{
{body}
    }}
}}
'''


def cuda_host_template(kernel_name, params, array_size=1024):
    """Generate host code for kernel launch."""
    
    # Generate memory allocation
    alloc_code = []
    for pname, ptype in params:
        alloc_code.append(f"    {ptype} *d_{pname};")
        alloc_code.append(f"    cudaMalloc(&d_{pname}, n * sizeof({ptype}));")
    
    alloc_str = "\n".join(alloc_code)
    
    # Generate kernel launch params
    param_list = ", ".join([f"d_{pname}" for pname, _ in params] + ["n"])
    
    # Generate memory cleanup
    free_code = "\n".join([f"    cudaFree(d_{pname});" for pname, _ in params])
    
    return f'''
// Host code for {kernel_name}
void launch_{kernel_name}(int n)
{{
    // Allocate device memory
{alloc_str}
    
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    {kernel_name}_kernel<<<numBlocks, blockSize>>>({param_list});
    
    // Wait for completion
    cudaDeviceSynchronize();
    
    // Free memory
{free_code}
}}
'''


def cuda_main_template(kernel_name):
    """Generate main function."""
    return f'''
int main()
{{
    int n = 1024;
    
    printf("Launching {kernel_name} kernel with %d elements\\n", n);
    launch_{kernel_name}(n);
    
    printf("Kernel completed successfully\\n");
    return 0;
}}
'''


def complete_cuda_file(kernel_name, params, body):
    """Generate complete .cu file."""
    
    header = '''
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \\
    do { \\
        cudaError_t err = call; \\
        if (err != cudaSuccess) { \\
            fprintf(stderr, "CUDA error in %s:%d: %s\\n", __FILE__, __LINE__, \\
                    cudaGetErrorString(err)); \\
            exit(EXIT_FAILURE); \\
        } \\
    } while(0)
'''
    
    kernel = cuda_kernel_template(kernel_name, params, body)
    host = cuda_host_template(kernel_name, params)
    main = cuda_main_template(kernel_name)
    
    return header + kernel + host + main


# Type mapping from Python/Warp to CUDA C++
TYPE_MAP = {
    "float": "float",
    "int": "int",
    "wp.float32": "float",
    "wp.int32": "int",
    "wp.vec3": "float3",
    "wp.vec2": "float2",
    "wp.vec4": "float4",
}


# Operation mapping
OP_MAP = {
    "wp.sin": "sinf",
    "wp.cos": "cosf",
    "wp.exp": "expf",
    "wp.sqrt": "sqrtf",
    "wp.abs": "fabsf",
    "wp.log": "logf",
    "wp.min": "fminf",
    "wp.max": "fmaxf",
    "+": "+",
    "-": "-",
    "*": "*",
    "/": "/",
}


def makefile_template(kernel_name):
    """Generate Makefile for compilation."""
    return f'''
# Makefile for {kernel_name}

NVCC = nvcc
NVCC_FLAGS = -arch=sm_50 -O3

# Target for CUDA code
{kernel_name}: {kernel_name}.cu
\t$(NVCC) $(NVCC_FLAGS) $< -o $@

# Generate PTX assembly
{kernel_name}.ptx: {kernel_name}.cu
\t$(NVCC) $(NVCC_FLAGS) --ptx $< -o $@

# Clean
clean:
\trm -f {kernel_name} {kernel_name}.ptx

.PHONY: clean
'''
