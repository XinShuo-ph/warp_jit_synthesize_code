"""IR Extraction utilities for Warp kernels."""
import os
from pathlib import Path
from dataclasses import dataclass

import warp as wp


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


def extract_ir(kernel: wp.Kernel, device: str = "cpu") -> IRPair:
    """
    Extract Python source and C++ IR from a warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        device: Device to compile for ("cpu" or "cuda")
    
    Returns:
        IRPair containing Python source and C++ IR
    """
    # Get Python source from adjoint
    python_source = kernel.adj.source
    
    # Force module compilation if not already compiled
    module = kernel.module
    module.load(device)
    
    # Get module identifier and locate cache
    module_id = module.get_module_identifier()
    cache_dir = get_cache_dir() / module_id
    
    # Determine file extension and kernel suffix based on device
    if device == "cpu":
        ext = ".cpp"
        suffix = "_cpu_kernel_"
    else:
        ext = ".cu"
        suffix = "_cuda_kernel_"

    # Find the generated file
    source_file = cache_dir / f"{module_id}{ext}"
    
    if not source_file.exists():
        raise FileNotFoundError(f"Generated source not found at {source_file}")
    
    source_code = source_file.read_text()
    
    return IRPair(
        kernel_name=kernel.key,
        python_source=python_source,
        cpp_ir=source_code,
        module_id=module_id
    )


def extract_kernel_functions(source_code: str, kernel_name: str, device: str = "cpu") -> dict:
    """
    Extract just the forward and backward functions for a specific kernel.
    
    Returns dict with 'forward' and 'backward' function bodies.
    """
    import re
    
    kernel_suffix = "_cpu_kernel_" if device == "cpu" else "_cuda_kernel_"
    
    # Use more flexible regex for nested braces
    result = {}
    
    # Find forward function
    forward_match = re.search(
        rf'(void\s+{re.escape(kernel_name)}_[a-f0-9]+{kernel_suffix}forward\([^)]*\)\s*\{{)',
        source_code
    )
    if forward_match:
        start = forward_match.start()
        # Count braces to find matching close brace
        brace_count = 0
        end = start
        for i, c in enumerate(source_code[start:]):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break
        result['forward'] = source_code[start:end]
    
    # Find backward function
    backward_match = re.search(
        rf'(void\s+{re.escape(kernel_name)}_[a-f0-9]+{kernel_suffix}backward\([^)]*\)\s*\{{)',
        source_code
    )
    if backward_match:
        start = backward_match.start()
        brace_count = 0
        end = start
        for i, c in enumerate(source_code[start:]):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break
        result['backward'] = source_code[start:end]
    
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
