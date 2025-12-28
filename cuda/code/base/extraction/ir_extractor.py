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
    
    # Find the generated C++ file
    cpp_file = cache_dir / f"{module_id}.cpp"
    
    if not cpp_file.exists():
        raise FileNotFoundError(f"Generated C++ not found at {cpp_file}")
    
    cpp_ir = cpp_file.read_text()
    
    return IRPair(
        kernel_name=kernel.key,
        python_source=python_source,
        cpp_ir=cpp_ir,
        module_id=module_id
    )


def extract_kernel_functions(cpp_ir: str, kernel_name: str) -> dict:
    """
    Extract just the forward and backward functions for a specific kernel.
    
    Returns dict with 'forward' and 'backward' C++ function bodies.
    """
    import re
    
    # Find the mangled kernel name (includes hash suffix)
    # Pattern: kernel_name_HASH_cpu_kernel_forward
    forward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_forward\([^)]*\)\s*\{{[^}}]*\}}'
    backward_pattern = rf'void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_backward\([^)]*\)\s*\{{[^}}]*\}}'
    
    # Use more flexible regex for nested braces
    result = {}
    
    # Find forward function
    forward_match = re.search(
        rf'(void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_forward\([^)]*\)\s*\{{)',
        cpp_ir
    )
    if forward_match:
        start = forward_match.start()
        # Count braces to find matching close brace
        brace_count = 0
        end = start
        for i, c in enumerate(cpp_ir[start:]):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break
        result['forward'] = cpp_ir[start:end]
    
    # Find backward function
    backward_match = re.search(
        rf'(void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_backward\([^)]*\)\s*\{{)',
        cpp_ir
    )
    if backward_match:
        start = backward_match.start()
        brace_count = 0
        end = start
        for i, c in enumerate(cpp_ir[start:]):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = start + i + 1
                    break
        result['backward'] = cpp_ir[start:end]
    
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
