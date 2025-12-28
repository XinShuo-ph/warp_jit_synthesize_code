"""
CUDA Code Generation Without GPU

This module generates CUDA IR code without requiring GPU hardware by leveraging
warp's internal codegen APIs. The key insight is that warp's ModuleBuilder can
generate CUDA code after the module is loaded for CPU - the internal AST and
adjoint structures are device-agnostic.

Usage:
    from cuda_codegen import generate_cuda_ir, extract_cuda_functions
    
    # Generate CUDA code for a kernel
    cuda_code = generate_cuda_ir(kernel)
    
    # Extract forward/backward functions
    funcs = extract_cuda_functions(cuda_code, kernel.key)
"""
import re
from dataclasses import dataclass
from typing import Optional

import warp as wp
from warp._src import context


@dataclass
class CUDAIRPair:
    """Holds Python source and generated CUDA IR for a kernel."""
    kernel_name: str
    kernel_key: str
    python_source: str
    cuda_ir_full: str
    cuda_ir_forward: str
    cuda_ir_backward: str
    

def generate_cuda_ir(kernel: wp.Kernel, options: Optional[dict] = None) -> str:
    """
    Generate CUDA IR from a kernel without requiring GPU hardware.
    
    The approach:
    1. Load the module for CPU first (populates internal structures)
    2. Use ModuleBuilder to generate CUDA code via codegen('cuda')
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        options: Optional codegen options (mode, block_dim, etc.)
    
    Returns:
        Generated CUDA source code as string
    """
    # Ensure module is loaded for CPU to populate internal structures
    kernel.module.load('cpu')
    
    # Set default options
    if options is None:
        options = {}
    options.setdefault('mode', 'release')
    options.setdefault('block_dim', 256)
    options.setdefault('enable_backward', True)
    
    # Create module builder
    hasher = context.ModuleHasher(kernel.module)
    builder = context.ModuleBuilder(kernel.module, options, hasher)
    
    # Build the kernel
    builder.build_kernel(kernel)
    
    # Generate CUDA code
    cuda_code = builder.codegen('cuda')
    
    return cuda_code


def extract_cuda_functions(cuda_ir: str, kernel_key: str) -> dict:
    """
    Extract forward and backward CUDA kernel functions.
    
    Args:
        cuda_ir: Full CUDA IR source code
        kernel_key: The kernel's key (used in function name mangling)
    
    Returns:
        Dict with 'forward' and 'backward' function bodies
    """
    result = {'forward': '', 'backward': ''}
    
    def extract_function(pattern: str, code: str) -> str:
        """Extract function body using brace counting."""
        match = re.search(pattern, code)
        if not match:
            return ""
        
        start = match.start()
        brace_count = 0
        end = start
        for i, c in enumerate(code[start:]):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break
        return code[start:end]
    
    # CUDA forward pattern: extern "C" __global__ void kernel_HASH_cuda_kernel_forward(...)
    forward_pattern = rf'(?:extern\s+"C"\s+)?__global__\s+void\s+{re.escape(kernel_key)}_[a-f0-9]+_cuda_kernel_forward\s*\([^)]*\)\s*\{{'
    result['forward'] = extract_function(forward_pattern, cuda_ir)
    
    # CUDA backward pattern
    backward_pattern = rf'(?:extern\s+"C"\s+)?__global__\s+void\s+{re.escape(kernel_key)}_[a-f0-9]+_cuda_kernel_backward\s*\([^)]*\)\s*\{{'
    result['backward'] = extract_function(backward_pattern, cuda_ir)
    
    return result


def generate_cuda_ir_pair(kernel: wp.Kernel, options: Optional[dict] = None) -> CUDAIRPair:
    """
    Generate a complete CUDA IR pair from a kernel.
    
    Args:
        kernel: A warp kernel
        options: Optional codegen options
    
    Returns:
        CUDAIRPair with Python source and CUDA IR
    """
    # Get Python source
    python_source = kernel.adj.source
    
    # Generate CUDA IR
    cuda_ir = generate_cuda_ir(kernel, options)
    
    # Extract functions
    funcs = extract_cuda_functions(cuda_ir, kernel.key)
    
    return CUDAIRPair(
        kernel_name=kernel.key.split('.')[-1] if '.' in kernel.key else kernel.key,
        kernel_key=kernel.key,
        python_source=python_source,
        cuda_ir_full=cuda_ir,
        cuda_ir_forward=funcs['forward'],
        cuda_ir_backward=funcs['backward']
    )


def validate_cuda_ir(cuda_ir: str) -> dict:
    """
    Validate that CUDA IR contains expected patterns.
    
    Returns dict with validation results.
    """
    patterns = {
        'has_global_decorator': r'__global__\s+void',
        'has_cuda_forward': r'_cuda_kernel_forward',
        'has_cuda_backward': r'_cuda_kernel_backward',
        'has_thread_indexing': r'blockDim\.x.*blockIdx\.x.*threadIdx\.x',
        'has_tile_shared_storage': r'wp::tile_shared_storage_t',
        'has_grid_stride_loop': r'for\s*\([^;]+;\s*_idx\s*<\s*dim\.size',
        'has_cuda_header': r'#define\s+__debugbreak',
    }
    
    results = {}
    for name, pattern in patterns.items():
        results[name] = bool(re.search(pattern, cuda_ir))
    
    results['all_valid'] = all(results.values())
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate CUDA IR without GPU")
    parser.add_argument("--validate", action="store_true", help="Validate output patterns")
    args = parser.parse_args()
    
    # Initialize warp
    wp.init()
    
    # Define a test kernel
    @wp.kernel
    def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] + b[tid] * 2.0
    
    print("=== CUDA Code Generation Without GPU ===\n")
    
    # Generate CUDA IR
    cuda_ir = generate_cuda_ir(test_kernel)
    
    print(f"Generated CUDA IR: {len(cuda_ir)} characters\n")
    
    if args.validate:
        print("=== Validation Results ===")
        validation = validate_cuda_ir(cuda_ir)
        for key, value in validation.items():
            status = "✓" if value else "✗"
            print(f"  {status} {key}: {value}")
        print()
    
    # Extract and show functions
    funcs = extract_cuda_functions(cuda_ir, test_kernel.key)
    
    print("=== Forward Function ===")
    print(funcs['forward'][:1000] if funcs['forward'] else "(not found)")
    
    print("\n=== Backward Function ===")
    print(funcs['backward'][:500] if funcs['backward'] else "(not found)")
