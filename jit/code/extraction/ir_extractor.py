"""IR Extractor: Extract Python source and generated C++ code from Warp kernels."""
import warp as wp
import warp._src.codegen
import warp._src.context
import re
from typing import Any


def extract_ir(kernel, device: str = "cpu", include_backward: bool = True) -> dict[str, Any]:
    """
    Extract IR (generated C++ code) from a Warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        device: Target device ("cpu" or "cuda")
        include_backward: If True, include backward/adjoint kernels
        
    Returns:
        dict with keys:
            - python_source: Original Python kernel source
            - cpp_code: Generated C++ code (full module)
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
    
    # Generate the full C++ source
    cpp_code = builder.codegen(device)
    
    # Extract Python source
    python_source = kernel.adj.source
    
    # Get kernel mangled name for extraction
    kernel_name = kernel.key
    mangled_name = kernel.get_mangled_name()
    
    # Extract just the forward kernel function
    forward_code = _extract_function(cpp_code, f"{mangled_name}_{device}_kernel_forward")
    
    # Extract backward kernel if requested
    backward_code = None
    if include_backward:
        backward_code = _extract_function(cpp_code, f"{mangled_name}_{device}_kernel_backward")
    
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
        # Device-specific generated source (CPU C++ or CUDA C++/CUDA)
        "generated_code": cpp_code,
        # Back-compat key name (historically called cpp_code even for CUDA)
        "cpp_code": cpp_code,
        "kernel_name": kernel_name,
        "forward_code": forward_code,
        "backward_code": backward_code,
        "metadata": metadata,
    }


def _extract_function(code: str, func_name: str) -> str | None:
    """Extract a single function from the generated code."""
    # Pattern to find function definition and its body
    # Match: void func_name(...) { ... }
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
    Simple extraction returning just the Python source and forward C++ code.
    
    Returns:
        (python_source, cpp_forward_code)
    """
    result = extract_ir(kernel, device, include_backward=False)
    return result["python_source"], result["forward_code"]


# Convenience function to extract multiple kernels
def extract_module_kernels(module) -> list[dict[str, Any]]:
    """Extract IR for all kernels in a module."""
    results = []
    for kernel in module.kernels.values():
        try:
            results.append(extract_ir(kernel))
        except Exception as e:
            results.append({
                "kernel_name": kernel.key,
                "error": str(e)
            })
    return results
