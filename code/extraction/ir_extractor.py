"""IR Extraction utilities for Warp kernels."""
import os
from pathlib import Path
from dataclasses import dataclass

import warp as wp

from code.common.device import resolve_warp_device


@dataclass
class IRPair:
    """Holds Python source and generated C++ IR for a kernel."""
    kernel_name: str
    python_source: str
    cpp_ir: str
    module_id: str


def get_cache_dir() -> Path:
    """Get the warp kernel cache directory."""
    return Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))

def get_generated_source_path(module_id: str, device: str) -> Path:
    """
    Locate the generated source file for a compiled Warp module.

    For CPU, Warp emits a `.cpp`.
    For CUDA, Warp typically emits a `.cu` (and may also emit host `.cpp`).
    """
    cache_dir = get_cache_dir() / module_id

    if device == "cpu":
        candidates = [cache_dir / f"{module_id}.cpp"]
    else:
        # Prefer CUDA source when targeting CUDA.
        candidates = [
            cache_dir / f"{module_id}.cu",
            cache_dir / f"{module_id}.cpp",
        ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Generated source not found for module '{module_id}' on device '{device}'. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def extract_ir(kernel: wp.Kernel, device: str = "cpu") -> IRPair:
    """
    Extract Python source and C++ IR from a warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        device: Device to compile for ("cpu" or "cuda")
    
    Returns:
        IRPair containing Python source and C++ IR
    """
    resolved = resolve_warp_device(device)

    # Get Python source from adjoint
    python_source = kernel.adj.source
    
    # Force module compilation if not already compiled
    module = kernel.module
    module.load(resolved.name)
    
    # Get module identifier and locate cache
    module_id = module.get_module_identifier()
    source_file = get_generated_source_path(module_id, resolved.name)
    cpp_ir = source_file.read_text()
    
    return IRPair(
        kernel_name=kernel.key,
        python_source=python_source,
        cpp_ir=cpp_ir,
        module_id=module_id
    )


def extract_kernel_functions(cpp_ir: str, kernel_name: str, device: str = "cpu") -> dict:
    """
    Extract just the forward and backward functions for a specific kernel.
    
    Returns dict with 'forward' and 'backward' C++ function bodies.
    """
    import re

    resolved = resolve_warp_device(device)
    
    def extract_function(start_pattern: str) -> str:
        match = re.search(start_pattern, cpp_ir)
        if not match:
            return ""
    
        start = match.start()
        brace_count = 0
        end = start
        for i, c in enumerate(cpp_ir[start:]):
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break
        return cpp_ir[start:end]
    
    # Some CUDA kernels may be emitted with `__global__ void ...`.
    # Function name suffix differs by target: `_cpu_kernel_*` vs `_cuda_kernel_*`.
    kernel_name_esc = re.escape(kernel_name)
    device_suffix = re.escape(resolved.name)

    forward_start = rf'((?:__global__\s+)?void\s+{kernel_name_esc}_[a-f0-9]+_{device_suffix}_kernel_forward\([^)]*\)\s*\{{)'
    backward_start = rf'((?:__global__\s+)?void\s+{kernel_name_esc}_[a-f0-9]+_{device_suffix}_kernel_backward\([^)]*\)\s*\{{)'

    result = {
        "forward": extract_function(forward_start),
        "backward": extract_function(backward_start),
    }
    
    return result


def extract_all_kernels_from_module(module: wp.context.Module, device: str = "cpu") -> list[IRPair]:
    """Extract IR pairs for all kernels in a module."""
    results = []
    for kernel in module.kernels.values():
        try:
            ir_pair = extract_ir(kernel, device)
            results.append(ir_pair)
        except Exception as e:
            print(f"Warning: Failed to extract IR for {kernel.key}: {e}")
    return results


if __name__ == "__main__":
    # Test the extractor
    wp.init()
    
    @wp.kernel
    def test_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] + b[tid]
    
    ir_pair = extract_ir(test_add)
    print("=== Python Source ===")
    print(ir_pair.python_source)
    print("\n=== C++ IR (first 1000 chars) ===")
    print(ir_pair.cpp_ir[:1000])
    print("\n=== Extracted Functions ===")
    funcs = extract_kernel_functions(ir_pair.cpp_ir, "test_add")
    for name, code in funcs.items():
        print(f"\n--- {name} ---")
        print(code[:500])
