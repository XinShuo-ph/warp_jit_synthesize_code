"""JAX IR Extractor - Extracts compiled code (HLO/LLVM/PTX) from JAX functions."""
import jax
import jax.extend.backend
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np

@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX kernel."""
    kernel_name: str
    python_source: str
    hlo_text: str
    llvm_ir: Optional[str] = None
    ptx_code: Optional[str] = None

def get_example_args(func_source: str):
    """Infer example arguments based on function signature (simple heuristic)."""
    # This is a bit hacky. We need to parse the source or use the function object 
    # to know how many args it expects.
    # We will assume args are arrays of shape (1024,) unless they look like scalars.
    # In JAX, we just pass shapes/dtypes to lower().
    
    # We'll use a standard shape for testing compilation
    N = 1024
    arr_shape = (N,)
    
    # Analyze source to guess arg count
    import re
    match = re.search(r'def \w+\((.*?)\):', func_source)
    if not match:
        return []
    
    arg_str = match.group(1)
    args = [a.strip() for a in arg_str.split(',')]
    
    example_args = []
    for arg in args:
        if arg in ['n', 'scale', 'alpha']:
            # Treat as scalar
            if arg == 'n':
                example_args.append(10) # int
            else:
                example_args.append(1.0) # float
        else:
            # Treat as array
            # For vector kernels, we might need vec3.
            # But our generator makes them generic or assumes last dim.
            # If the source contains 'dot' or 'norm', maybe 2D array?
            # Let's keep it simple: 1D array for elementwise.
            # For dot/norm in our generator, we used axis=-1.
            # If inputs are (N, 3), axis=-1 works for vec3.
            # If inputs are (N,), axis=-1 works (scalar dot).
            # Let's use (N, 3) if it seems like a vector kernel, else (N,).
            if 'vec' in func_source or 'dot' in func_source or 'norm' in func_source:
                 example_args.append(jnp.ones((N, 3)))
            else:
                 example_args.append(jnp.ones((N,)))
    
    return example_args

def extract_ir(func: Any, source_code: str) -> ExtractedIR:
    """
    Extract IR from a JAX function.
    
    Args:
        func: The JAX function (jitted or not)
        source_code: The source code string
        
    Returns:
        ExtractedIR
    """
    
    # Get example args to trace the function
    args = get_example_args(source_code)
    
    # Lower the function to get HLO
    # We use jax.jit to ensure it's compiled
    jit_func = jax.jit(func)
    
    lowered = jit_func.lower(*args)
    
    hlo_text = lowered.as_text()
    
    # Get compiled artifact to access LLVM/PTX
    compiled = lowered.compile()
    
    # Extract ASM (PTX for GPU, Assembly for CPU)
    # Note: compiled.as_text() might give assembly or something else depending on backend
    # For CPU, we want LLVM IR. For GPU, PTX.
    
    llvm_ir = None
    ptx_code = None
    
    backend = jax.extend.backend.get_backend().platform
    
    # JAX exposes different things.
    # compiled.as_text() usually returns the assembly.
    # To get LLVM IR, we might need to look at the HLO module and compile it with XLA manually 
    # or use private APIs.
    # However, `lowered.as_text("hlo")` gives HLO.
    # `compiled.as_text()` gives the assembly of the machine code.
    
    # Let's just store what we can easily get.
    # "cpp_code" in Warp was LLVM-like C++ or actual C++.
    # Here we will map:
    # cpp_code -> LLVM IR (or HLO if LLVM not available easily)
    # cuda_code -> PTX
    
    # Actually, for the purpose of the dataset, HLO is the most "intermediate" representation.
    # But let's try to get lower level if possible.
    
    # On CPU, compiled.as_text() is likely x86 assembly.
    # On GPU, it's PTX.
    
    asm = compiled.as_text()
    
    if backend == 'cpu':
        # On CPU, let's treat the ASM as the "cpp_code" equivalent (low level)
        # But maybe the user wants LLVM IR?
        # compiled.modules[0].as_text() might exist? No.
        llvm_ir = asm
    elif backend == 'gpu' or backend == 'cuda':
        ptx_code = asm
    
    return ExtractedIR(
        kernel_name=getattr(func, '__name__', 'unknown'),
        python_source=source_code,
        hlo_text=hlo_text,
        llvm_ir=llvm_ir,
        ptx_code=ptx_code
    )
