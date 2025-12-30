"""IR Extractor - Extracts generated JAX HLO from kernels."""
from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
import inspect
import numpy as np

@dataclass
class ExtractedIR:
    """Container for extracted IR from a kernel."""
    kernel_name: str
    python_source: str
    cpp_code: str  # We will store HLO here
    cuda_code: Optional[str] = None  # We will store optimized HLO or backend specific code here if possible

def get_dummy_inputs(func_name, arg_names):
    """Generate dummy inputs based on argument names and kernel type."""
    N = 128
    inputs = []
    
    is_vector_kernel = "vec" in func_name
    
    for name in arg_names:
        if name in ["n"]:
            inputs.append(N)
        elif name in ["scale", "alpha", "threshold", "val", "const1", "const2"]:
            inputs.append(1.0) # float scalar
        elif name in ["a", "b", "c", "x", "y", "out", "result", "temp1", "temp2"]:
            if is_vector_kernel and name in ["a", "b"]:
                inputs.append(jnp.ones((N, 3), dtype=jnp.float32))
            else:
                inputs.append(jnp.ones((N,), dtype=jnp.float32))
        else:
            # Default to float array
            inputs.append(jnp.ones((N,), dtype=jnp.float32))
            
    return inputs

def extract_ir(kernel_func, enable_backward: bool = True) -> ExtractedIR:
    """
    Extract IR (HLO) from a JAX kernel.
    
    Args:
        kernel_func: A JAX jitted function
        enable_backward: Whether to include backward (adjoint) code (Not implemented for JAX HLO extraction yet in this simple script, usually handled by jax.grad)
        
    Returns:
        ExtractedIR containing Python source and generated HLO
    """
    
    # Unwrap jit to get original function for inspection if needed, 
    # but we need the jitted version for lowering.
    # jax.jit returns a Pjit object.
    
    # Get argument names
    # If it's a Pjit object, we might need to look at the wrapped function
    if hasattr(kernel_func, '__wrapped__'):
        orig_func = kernel_func.__wrapped__
    else:
        orig_func = kernel_func
        
    sig = inspect.signature(orig_func)
    arg_names = list(sig.parameters.keys())
    
    # Generate dummy inputs
    inputs = get_dummy_inputs(orig_func.__name__, arg_names)
    
    # Lower to HLO
    # We produce HLO text
    lowered = kernel_func.lower(*inputs)
    hlo_text = lowered.as_text()
    
    # If we want backend specific compilation:
    # compiled = lowered.compile()
    # asm = compiled.as_text() # This might be assembly
    
    # For now, let's put HLO in cpp_code
    cpp_code = hlo_text
    
    # If we want to simulate "CUDA code", we could try to compile for GPU if available,
    # or just leave it empty.
    # Or we can put the "optimized" HLO in cuda_code?
    cuda_code = None
    
    # Extract source
    source = inspect.getsource(orig_func)
    
    return ExtractedIR(
        kernel_name=orig_func.__name__,
        python_source=source,
        cpp_code=cpp_code,
        cuda_code=cuda_code,
    )


def extract_ir_pair(kernel, device: str = "cpu") -> tuple[str, str]:
    """
    Extract Python→HLO pair.
    
    Args:
        kernel: A JAX kernel
        device: "cpu" or "cuda" (ignored for HLO usually)
        
    Returns:
        Tuple of (python_source, code)
    """
    ir = extract_ir(kernel)
    return (ir.python_source, ir.cpp_code)


if __name__ == "__main__":
    # Test with a simple kernel
    
    @jax.jit
    def test_kernel(a, b):
        return a * 2.0 + b
    
    ir = extract_ir(test_kernel)
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== HLO Code (first 1500 chars) ===")
    print(ir.cpp_code[:1500])
