import warp as wp
import sys
import os
from typing import Dict, Any, Optional

# Import internal Warp modules
# Note: accessing _src is technically internal API, but necessary for this task
from warp._src.context import ModuleBuilder
from warp._src.codegen import codegen_func

def get_kernel_ir(kernel: wp.kernel, device: str = "cpu") -> Dict[str, str]:
    """
    Extracts the Intermediate Representation (C++ source) from a Warp kernel.
    
    Args:
        kernel: The warp kernel object (decorated with @wp.kernel)
        device: Target device ("cpu" or "cuda"). Defaults to "cpu".
        
    Returns:
        A dictionary containing the generated source code segments.
        Keys:
            - 'forward': The C++ code for the forward pass
            - 'backward': The C++ code for the backward pass (if applicable)
            - 'source': The original Python source code
    """
    if not isinstance(kernel, wp.context.Kernel):
        raise ValueError("Input must be a warp kernel")

    # Use default options compatible with the current warp version
    options = {
        "max_unroll": 4,
        "enable_backward": True, # Enable to allow backward generation check
        "fast_math": False,
        "fuse_fp": True,
        "lineinfo": False,
        "cuda_output": None,
        "mode": device,
        "optimization_level": 3,
        "block_dim": 256,
        "compile_time_trace": False,
        "strip_hash": False,
    }

    # Ensure the Adjoint is built
    # We use the kernel's existing module or a fresh builder context
    # Note: We need to be careful not to corrupt the kernel's state if it's being used elsewhere,
    # but build() should be idempotent-ish or cumulative.
    
    # We create a Builder to trigger the AST traversal and block population
    builder = ModuleBuilder(kernel.module, options)
    
    # Check if already built? 
    # adj.return_var being None usually means not fully built or void return (which kernels are).
    # But adj.blocks being empty means definitely not built.
    if not hasattr(kernel.adj, 'blocks') or not kernel.adj.blocks:
        kernel.adj.build(builder)
    
    # Generate Forward Pass
    forward_code = codegen_func(
        kernel.adj,
        c_func_name=f"{kernel.key}_forward",
        device=device,
        options=options,
        forward_only=True
    )
    
    # Generate Backward Pass
    # Only if backward is enabled/possible
    backward_code = ""
    try:
        backward_code = codegen_func(
            kernel.adj,
            c_func_name=f"{kernel.key}_backward",
            device=device,
            options=options,
            reverse_only=True
        )
    except Exception:
        # Some kernels might not support backward or it might fail
        pass

    result = {
        "forward": forward_code,
        "backward": backward_code,
        "source": kernel.adj.source
    }
    
    return result
