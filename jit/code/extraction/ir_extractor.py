"""IR Extractor: Extract Python source and generated C++/CUDA code from Warp kernels."""
import warp as wp
import warp._src.codegen
import warp._src.context
import re
from typing import Any


def extract_ir(kernel, device: str = "cpu", include_backward: bool = True) -> dict[str, Any]:
    """
    Extract IR (generated C++/CUDA code) from a Warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        device: Target device ("cpu" or "cuda")
        include_backward: If True, include backward/adjoint kernels
        
    Returns:
        dict with keys:
            - python_source: Original Python kernel source
            - cpp_code: Generated C++/CUDA code (full module)
            - kernel_name: Name of the kernel
            - forward_code: Just the forward kernel function
            - backward_code: Just the backward kernel function (if include_backward)
            - metadata: Dict with kernel info (arg names, types, etc.)
    """
    module = kernel.module
    
    # Get or create hasher for the module
    hasher = warp._src.context.ModuleHasher(module)
    
    # Create options dict
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", include_backward)
    options.setdefault("mode", "release")
    
    # Create a builder to generate code
    builder = warp._src.context.ModuleBuilder(module, options, hasher)
    
    # Generate the full C++/CUDA source
    cpp_code = builder.codegen(device)
    
    # Extract Python source
    python_source = kernel.adj.source
    
    # Get kernel mangled name for extraction
    kernel_name = kernel.key
    mangled_name = kernel.get_mangled_name()
    
    # Extract just the forward kernel function
    forward_code = _extract_function(cpp_code, f"{mangled_name}_{device}_kernel_forward", device)
    
    # Extract backward kernel if requested
    backward_code = None
    if include_backward:
        backward_code = _extract_function(cpp_code, f"{mangled_name}_{device}_kernel_backward", device)
    
    # Build metadata
    metadata = {
        "kernel_name": kernel_name,
        "mangled_name": mangled_name,
        "device": device,
        "arg_names": list(kernel.adj.arg_types.keys()),
        "arg_types": {k: str(v) for k, v in kernel.adj.arg_types.items()},
        "has_backward": backward_code is not None,
    }
    
    return {
        "python_source": python_source,
        "cpp_code": cpp_code,
        "kernel_name": kernel_name,
        "forward_code": forward_code,
        "backward_code": backward_code,
        "metadata": metadata,
    }


def _extract_function(code: str, func_name: str, device: str = "cpu") -> str | None:
    """Extract a single function from the generated code."""
    # Pattern differs for CPU vs CUDA
    if device == "cuda":
        pattern = rf'__global__ void {re.escape(func_name)}\s*\([^)]*\)\s*\{{'
    else:
        pattern = rf'void {re.escape(func_name)}\s*\([^)]*\)\s*\{{'
    
    match = re.search(pattern, code)
    if not match:
        return None
    
    start = match.start()
    
    # Find matching closing brace
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


def extract_python_ir_pair(kernel, device: str = "cpu") -> tuple[str, str]:
    """
    Simple extraction returning just the Python source and forward IR code.
    
    Returns:
        (python_source, ir_forward_code)
    """
    result = extract_ir(kernel, device, include_backward=False)
    return result["python_source"], result["forward_code"]


def extract_for_training(kernel, device: str = "cpu") -> dict[str, Any] | None:
    """
    Extract data suitable for LLM training.
    
    Returns dict with:
        - python_source: The Python kernel code
        - ir_code: The generated C++/CUDA forward function
        - metadata: Kernel metadata
    """
    try:
        result = extract_ir(kernel, device, include_backward=False)
        
        if not result["forward_code"]:
            return None
        
        return {
            "python_source": result["python_source"],
            "ir_code": result["forward_code"],
            "metadata": result["metadata"]
        }
    except Exception as e:
        return None


if __name__ == "__main__":
    # Demo extraction
    @wp.kernel
    def example_kernel(
        a: wp.array(dtype=float),
        b: wp.array(dtype=float),
        out: wp.array(dtype=float)
    ):
        tid = wp.tid()
        out[tid] = a[tid] + b[tid]
    
    wp.init()
    
    print("=== CPU IR ===")
    result = extract_ir(example_kernel, device="cpu")
    print(f"Kernel: {result['kernel_name']}")
    print(f"Forward function:\n{result['forward_code']}")
    
    print("\n=== CUDA IR ===")
    result = extract_ir(example_kernel, device="cuda")
    print(f"Kernel: {result['kernel_name']}")
    print(f"Forward function:\n{result['forward_code']}")
