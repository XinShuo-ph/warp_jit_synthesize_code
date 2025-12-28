"""IR Extractor: Extract Python source and generated C++ code from Warp kernels."""
import warp as wp
import warp._src.codegen
import warp._src.context
import re
from typing import Any


def _normalize_device(device: str) -> str:
    device = (device or "").lower().strip()
    if device in {"cpu", "cuda"}:
        return device
    raise ValueError(f"Unsupported device: {device!r} (expected 'cpu' or 'cuda')")


def _ensure_device_available(device: str) -> None:
    """Fail fast with a helpful message if a device is not usable."""
    # wp.init() is idempotent; it will print a warning if CUDA is missing.
    wp.init()
    aliases: set[str] = set()
    for d in wp.get_devices():
        if isinstance(d, str):
            aliases.add(d)
        else:
            aliases.add(getattr(d, "alias", str(d)))

    if device not in aliases:
        if device == "cuda":
            raise RuntimeError(
                "CUDA device not available (warp reports no 'cuda' device). "
                "On a GPU machine, install a compatible NVIDIA driver/CUDA runtime and retry."
            )
        raise RuntimeError(f"Device {device!r} not available (warp devices={sorted(aliases)!r})")


def _code_extension(device: str) -> str:
    return ".cu" if device == "cuda" else ".cpp"


def extract_ir(
    kernel,
    device: str = "cpu",
    include_backward: bool = True,
    *,
    require_device: bool = True,
) -> dict[str, Any]:
    """
    Extract IR (generated C++ code) from a Warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        device: Target device ("cpu" or "cuda")
        include_backward: If True, include backward/adjoint kernels
        require_device: If False, allow generating CUDA source even if no CUDA device is available.
            This supports a "CUDA code-only" pipeline that does not launch kernels.
        
    Returns:
        dict with keys:
            - python_source: Original Python kernel source
            - cpp_code: Generated C++ code (full module)
            - kernel_name: Name of the kernel
            - forward_code: Just the forward kernel function
            - backward_code: Just the backward kernel function (if include_backward)
            - metadata: Dict with kernel info (arg names, types, etc.)
    """
    device = _normalize_device(device)
    if require_device:
        _ensure_device_available(device)
    else:
        # Still initialize Warp runtime; codegen('cuda') can succeed without a CUDA driver.
        wp.init()

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
    
    # Generate the full source for the chosen backend.
    # For CUDA this is CUDA C/C++; for CPU this is C++.
    generated_code = builder.codegen(device)
    
    # Extract Python source
    python_source = kernel.adj.source
    
    # Get kernel mangled name for extraction
    kernel_name = kernel.key
    mangled_name = kernel.get_mangled_name()
    
    # Extract just the forward kernel function
    forward_code = _extract_function(generated_code, f"{mangled_name}_{device}_kernel_forward")
    
    # Extract backward kernel if requested
    backward_code = None
    if include_backward:
        backward_code = _extract_function(generated_code, f"{mangled_name}_{device}_kernel_backward")
    
    # Build metadata
    metadata = {
        "kernel_name": kernel_name,
        "mangled_name": mangled_name,
        "device": device,
        "code_ext": _code_extension(device),
        "arg_names": list(kernel.adj.arg_types.keys()),
        "arg_types": {k: str(v) for k, v in kernel.adj.arg_types.items()},
        "has_backward": backward_code is not None,
    }
    
    return {
        "python_source": python_source,
        # Back-compat field name (historically CPU-only). For CUDA this contains CUDA code.
        "cpp_code": generated_code,
        "kernel_name": kernel_name,
        "forward_code": forward_code,
        "backward_code": backward_code,
        "metadata": metadata,
    }


def _extract_function(code: str, func_name: str) -> str | None:
    """Extract a single function from the generated code."""
    # Pattern to find function definition and its body.
    # CPU typically:          void func_name(...) { ... }
    # CUDA typically: __global__ void func_name(...) { ... }
    qualifiers = r"(?:__global__\s+|__host__\s+|__device__\s+|static\s+|inline\s+)*"
    pattern = rf'{qualifiers}void {re.escape(func_name)}\s*\([^)]*\)\s*\{{'
    
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
