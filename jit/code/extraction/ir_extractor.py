"""IR Extractor - Extracts generated C++/CUDA code from Warp kernels."""
from dataclasses import dataclass
from typing import Optional
import warp as wp
import warp._src.codegen
import warp._src.context


@dataclass
class ExtractedIR:
    """Container for extracted IR from a warp kernel."""
    kernel_name: str
    python_source: str
    cpp_code: str  # CPU code with forward + backward
    cuda_code: Optional[str] = None  # CUDA code with forward + backward


def extract_ir(kernel: wp.Kernel, enable_backward: bool = True) -> ExtractedIR:
    """
    Extract IR (C++/CUDA code) from a Warp kernel.
    
    Args:
        kernel: A warp kernel decorated with @wp.kernel
        enable_backward: Whether to include backward (adjoint) code
        
    Returns:
        ExtractedIR containing Python source and generated C++/CUDA code
    """
    # Ensure the kernel's adjoint is built
    if not hasattr(kernel, 'adj') or kernel.adj is None:
        raise ValueError("Kernel has no adjoint - ensure it's decorated with @wp.kernel")
    
    module = kernel.module
    
    # Get or create hasher for the module
    hasher = warp._src.context.ModuleHasher(module)
    
    # Create options dict
    options = module.options.copy() if module.options else {}
    options.setdefault("block_dim", 256)
    options.setdefault("enable_backward", enable_backward)
    options.setdefault("mode", "release")
    
    # Create a builder to generate code
    builder = warp._src.context.ModuleBuilder(module, options, hasher)
    
    # Generate CPU code (includes both forward and backward)
    cpp_code = builder.codegen("cpu")
    
    # Generate CUDA code (includes both forward and backward)
    cuda_code = None
    try:
        cuda_code = builder.codegen("cuda")
    except Exception:
        pass  # CUDA codegen may not be available
    
    return ExtractedIR(
        kernel_name=kernel.key,
        python_source=kernel.adj.source,
        cpp_code=cpp_code,
        cuda_code=cuda_code,
    )


def extract_ir_pair(kernel: wp.Kernel, device: str = "cpu") -> tuple[str, str]:
    """
    Extract Python→C++ pair suitable for LLM training.
    
    Args:
        kernel: A warp kernel
        device: "cpu" or "cuda"
        
    Returns:
        Tuple of (python_source, code)
    """
    ir = extract_ir(kernel)
    if device == "cuda" and ir.cuda_code:
        return (ir.python_source, ir.cuda_code)
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
    print("\n=== CUDA Code available ===")
    print("Yes" if ir.cuda_code else "No")
    if ir.cuda_code:
        print("\n=== CUDA Code (first 1500 chars) ===")
        print(ir.cuda_code[:1500])
