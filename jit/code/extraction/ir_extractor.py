"""IR Extractor - Extracts generated C++/CUDA code from Warp kernels."""
from dataclasses import dataclass
from typing import Optional
import warp as wp
from warp._src.codegen import codegen_kernel, codegen_module


@dataclass
class ExtractedIR:
    """Container for extracted IR from a warp kernel."""
    kernel_name: str
    python_source: str
    cpp_code: str
    cuda_code: Optional[str] = None


def extract_ir(kernel: wp.Kernel, enable_backward: bool = True, device: str = "cpu") -> ExtractedIR:
    """
    Extract IR (C++/CUDA code) from a Warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        enable_backward: Whether to include backward (adjoint) code
        device: Target device for compilation ("cpu" or "cuda")
        
    Returns:
        ExtractedIR containing Python source and generated C++/CUDA code
    """
    # Ensure the kernel's adjoint is built
    if not hasattr(kernel, 'adj') or kernel.adj is None:
        raise ValueError("Kernel has no adjoint - ensure it's decorated with @wp.kernel")
    
    # Load the module to trigger hash computation and full build
    kernel.module.load(device)
    
    options = {
        "enable_backward": enable_backward,
        "mode": "release",
    }
    # Merge with module options
    options = kernel.module.options | options
    
    # Generate CPU code
    cpp_kernel = codegen_kernel(kernel, "cpu", options)
    cpp_module = codegen_module(kernel, "cpu", options)
    cpp_code = cpp_kernel + "\n" + cpp_module
    
    # Try to generate CUDA code (may fail on CPU-only systems)
    cuda_code = None
    try:
        cuda_kernel = codegen_kernel(kernel, "cuda", options)
        cuda_code = cuda_kernel
    except Exception:
        pass  # CUDA codegen may not be available
    
    return ExtractedIR(
        kernel_name=kernel.key,
        python_source=kernel.adj.source,
        cpp_code=cpp_code,
        cuda_code=cuda_code,
    )


def extract_ir_pair(kernel: wp.Kernel) -> tuple[str, str]:
    """
    Extract Python→C++ pair suitable for LLM training.
    
    Args:
        kernel: A warp kernel
        
    Returns:
        Tuple of (python_source, cpp_code)
    """
    ir = extract_ir(kernel)
    return (ir.python_source, ir.cpp_code)


if __name__ == "__main__":
    # Test with a simple kernel
    wp.init()
    
    @wp.kernel
    def test_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
        tid = wp.tid()
        b[tid] = a[tid] * 2.0
    
    ir = extract_ir(test_kernel)
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== C++ Code (first 1500 chars) ===")
    print(ir.cpp_code[:1500])
