"""IR Extractor - Extracts generated HLO/XLA code from JAX functions."""
from dataclasses import dataclass
from typing import Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax._src.interpreters import mlir
from jax._src.lib import xla_client
import inspect


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    function_name: str
    python_source: str
    jaxpr_code: str  # JAX intermediate representation
    hlo_code: Optional[str] = None  # HLO (High-Level Optimizer) representation
    optimized_hlo_code: Optional[str] = None  # Optimized HLO
    

def extract_ir(func: Callable, *args, enable_backward: bool = True, **kwargs) -> ExtractedIR:
    """
    Extract IR (JAXPR/HLO code) from a JAX function.
    
    Args:
        func: A Python function to be JIT-compiled
        *args: Example inputs for the function
        enable_backward: Whether to include backward (gradient) code
        **kwargs: Additional keyword arguments for the function
        
    Returns:
        ExtractedIR containing Python source and generated JAXPR/HLO code
    """
    # Get Python source
    try:
        python_source = inspect.getsource(func)
    except:
        python_source = f"# Source not available for {func.__name__}"
    
    # Get JAXPR (JAX's intermediate representation)
    jaxpr = make_jaxpr(func)(*args, **kwargs)
    jaxpr_str = str(jaxpr)
    
    # Get HLO representation
    hlo_code = None
    optimized_hlo_code = None
    
    try:
        # Create a JIT-compiled version
        jitted_func = jax.jit(func)
        
        # Lower to get HLO
        lowered = jax.jit(func).lower(*args, **kwargs)
        
        # Get HLO text representation
        hlo_code = lowered.as_text()
        
        # Get optimized HLO (after compilation)
        compiled = lowered.compile()
        try:
            # Try to get optimized HLO from compiled object
            optimized_hlo_code = compiled.as_text()
        except:
            optimized_hlo_code = "# Optimized HLO not available"
            
    except Exception as e:
        hlo_code = f"# HLO extraction failed: {e}"
        optimized_hlo_code = "# Optimized HLO not available"
    
    # If backward is enabled, also extract gradient function IR
    if enable_backward:
        try:
            # Create gradient function
            grad_func = jax.grad(func) if len(args) > 0 else None
            if grad_func is not None:
                grad_jaxpr = make_jaxpr(grad_func)(*args, **kwargs)
                jaxpr_str += "\n\n# Backward (Gradient) JAXPR:\n" + str(grad_jaxpr)
                
                # Get HLO for gradient
                grad_lowered = jax.jit(grad_func).lower(*args, **kwargs)
                grad_hlo = grad_lowered.as_text()
                hlo_code += "\n\n# Backward (Gradient) HLO:\n" + grad_hlo
        except Exception as e:
            jaxpr_str += f"\n\n# Gradient extraction failed: {e}"
    
    return ExtractedIR(
        function_name=func.__name__,
        python_source=python_source,
        jaxpr_code=jaxpr_str,
        hlo_code=hlo_code,
        optimized_hlo_code=optimized_hlo_code,
    )


def extract_ir_pair(func: Callable, *args, device: str = "cpu", **kwargs) -> tuple[str, str]:
    """
    Extract Python→HLO pair suitable for LLM training.
    
    Args:
        func: A JAX function
        *args: Example inputs
        device: "cpu" or "gpu" (JAX automatically handles device placement)
        **kwargs: Additional keyword arguments
        
    Returns:
        Tuple of (python_source, ir_code)
    """
    ir = extract_ir(func, *args, **kwargs)
    # For JAX, we use HLO as the primary IR representation
    ir_code = ir.hlo_code if ir.hlo_code else ir.jaxpr_code
    return (ir.python_source, ir_code)


if __name__ == "__main__":
    # Test with a simple function
    def test_function(a, b):
        """Simple test function."""
        return a * 2.0 + b
    
    # Create example inputs
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    
    ir = extract_ir(test_function, a, b)
    print("=== Function Name ===")
    print(ir.function_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== JAXPR Code ===")
    print(ir.jaxpr_code)
    print("\n=== HLO Code available ===")
    print("Yes" if ir.hlo_code else "No")
    if ir.hlo_code and len(ir.hlo_code) < 2000:
        print("\n=== HLO Code ===")
        print(ir.hlo_code)
    elif ir.hlo_code:
        print("\n=== HLO Code (first 1500 chars) ===")
        print(ir.hlo_code[:1500])
