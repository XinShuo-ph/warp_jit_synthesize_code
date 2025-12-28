"""IR Extraction utilities for Warp kernels."""
import os
from pathlib import Path
from dataclasses import dataclass

import warp as wp


@dataclass
class IRPair:
    """Holds Python source and generated C++/CUDA IR for a kernel."""
    kernel_name: str
    python_source: str
    cpp_ir: str
    module_id: str
    device: str = "cpu"  # "cpu" or "cuda"


def get_cache_dir() -> Path:
    """Get the warp kernel cache directory."""
    return Path(os.path.expanduser(f"~/.cache/warp/{wp.__version__}"))


def is_cuda_available() -> bool:
    """Check if CUDA device is available."""
    try:
        devices = wp.get_devices()
        return any("cuda" in str(d) for d in devices)
    except Exception:
        return False


def extract_ir(kernel: wp.Kernel, device: str = "cpu") -> IRPair:
    """
    Extract Python source and C++/CUDA IR from a warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        device: Device to compile for ("cpu" or "cuda")
    
    Returns:
        IRPair containing Python source and C++/CUDA IR
        
    Raises:
        RuntimeError: If CUDA is requested but not available
    """
    # Check CUDA availability
    if device == "cuda" and not is_cuda_available():
        raise RuntimeError(
            "CUDA device requested but not available. "
            "CUDA code generation requires GPU hardware with CUDA driver. "
            "Use device='cpu' for CPU IR extraction."
        )
    
    # Get Python source from adjoint
    python_source = kernel.adj.source
    
    # Force module compilation if not already compiled
    module = kernel.module
    module.load(device)
    
    # Get module identifier and locate cache
    module_id = module.get_module_identifier()
    cache_dir = get_cache_dir() / module_id
    
    # Find the generated source file based on device
    if device == "cuda":
        # CUDA generates .cu file
        ir_file = cache_dir / f"{module_id}.cu"
        if not ir_file.exists():
            # Fallback: try .cpp if .cu not found (older warp versions)
            ir_file = cache_dir / f"{module_id}.cpp"
    else:
        # CPU generates .cpp file
        ir_file = cache_dir / f"{module_id}.cpp"
    
    if not ir_file.exists():
        raise FileNotFoundError(f"Generated IR not found at {ir_file}")
    
    ir_code = ir_file.read_text()
    
    return IRPair(
        kernel_name=kernel.key,
        python_source=python_source,
        cpp_ir=ir_code,
        module_id=module_id,
        device=device
    )


def extract_kernel_functions(ir_code: str, kernel_name: str, device: str = "cpu") -> dict:
    """
    Extract just the forward and backward functions for a specific kernel.
    
    Args:
        ir_code: The generated C++/CUDA source code
        kernel_name: Name of the kernel to extract
        device: "cpu" or "cuda" - affects pattern matching
    
    Returns dict with 'forward' and 'backward' function bodies.
    """
    import re
    
    result = {}
    
    # Device-specific patterns
    # CPU: void kernel_name_HASH_cpu_kernel_forward(...)
    # CUDA: extern "C" __global__ void kernel_name_HASH_cuda_kernel_forward(...)
    if device == "cuda":
        forward_pattern = rf'((?:extern\s+"C"\s+)?__global__\s+void\s+{re.escape(kernel_name)}_[a-f0-9]+_cuda_kernel_forward\([^)]*\)\s*\{{)'
        backward_pattern = rf'((?:extern\s+"C"\s+)?__global__\s+void\s+{re.escape(kernel_name)}_[a-f0-9]+_cuda_kernel_backward\([^)]*\)\s*\{{)'
    else:
        forward_pattern = rf'(void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_forward\([^)]*\)\s*\{{)'
        backward_pattern = rf'(void\s+{re.escape(kernel_name)}_[a-f0-9]+_cpu_kernel_backward\([^)]*\)\s*\{{)'
    
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
    
    result['forward'] = extract_function(forward_pattern, ir_code)
    result['backward'] = extract_function(backward_pattern, ir_code)
    
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Test IR extraction")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to compile for")
    args = parser.parse_args()
    
    # Test the extractor
    wp.init()
    
    @wp.kernel
    def test_add(a: wp.array(dtype=float), b: wp.array(dtype=float), c: wp.array(dtype=float)):
        tid = wp.tid()
        c[tid] = a[tid] + b[tid]
    
    device = args.device
    print(f"=== Testing IR Extraction for device: {device} ===\n")
    
    ir_pair = extract_ir(test_add, device=device)
    print("=== Python Source ===")
    print(ir_pair.python_source)
    print(f"\n=== {device.upper()} IR (first 1000 chars) ===")
    print(ir_pair.cpp_ir[:1000])
    print(f"\n=== Extracted Functions (device={device}) ===")
    funcs = extract_kernel_functions(ir_pair.cpp_ir, "test_add", device=device)
    for name, code in funcs.items():
        print(f"\n--- {name} ---")
        print(code[:500] if code else "(not found)")
