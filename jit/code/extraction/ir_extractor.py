"""IR Extractor - Extracts HLO from JAX kernels."""
from dataclasses import dataclass
from typing import Optional, Any
import jax
import jax.numpy as jnp
import inspect

@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX kernel."""
    kernel_name: str
    python_source: str
    hlo_code: str  # HLO text
    # Keeping these for compatibility with pipeline, but mapped to HLO or empty
    cpp_code: str  
    cuda_code: Optional[str] = None


def get_dummy_args(func, kernel_name: str):
    """Generate dummy arguments for JAX function based on signature and name."""
    sig = inspect.signature(func)
    args = []
    
    # Standard shapes
    N = 128
    
    is_vector_kernel = "vec" in kernel_name
    
    for param_name in sig.parameters:
        if param_name in ["n", "count"]:
            # Integer scalar
            args.append(10)
        elif param_name in ["alpha", "scale", "threshold"]:
            # Float scalar
            args.append(1.0)
        elif param_name in ["a", "b", "c", "x", "y", "result"]:
            if is_vector_kernel:
                # (N, 3) array
                args.append(jnp.ones((N, 3), dtype=jnp.float32))
            else:
                # (N,) array
                args.append(jnp.ones((N,), dtype=jnp.float32))
        else:
            # Default to array
            args.append(jnp.ones((N,), dtype=jnp.float32))
            
    return args


def extract_ir(func: Any, func_name: str = "unknown") -> ExtractedIR:
    """
    Extract HLO from a JAX function.
    
    Args:
        func: A JAX-jitted function
        func_name: Name of the function
        
    Returns:
        ExtractedIR containing Python source and generated HLO
    """
    
    try:
        args = get_dummy_args(func, func_name)
        
        # Lower to HLO
        hlo_text = func.lower(*args).as_text()
        
        # Helper to try to get source
        try:
            source = inspect.getsource(func)
        except:
            source = ""

        return ExtractedIR(
            kernel_name=func_name,
            python_source=source,
            hlo_code=hlo_text,
            cpp_code=hlo_text,  # Mapping HLO to cpp_code field for compatibility
            cuda_code=hlo_text  # Mapping HLO to cuda_code field for compatibility
        )
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract IR for {func_name}: {e}")


def extract_ir_pair(func: Any, device: str = "cpu") -> tuple[str, str]:
    """
    Extract Python->HLO pair.
    """
    # Just return HLO for both devices since HLO is high-level
    ir = extract_ir(func)
    return (ir.python_source, ir.hlo_code)


if __name__ == "__main__":
    # Test with a simple kernel
    @jax.jit
    def test_kernel(a, b):
        return a * b + 2.0
    
    ir = extract_ir(test_kernel, "test_kernel")
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== HLO Code (first 500 chars) ===")
    print(ir.hlo_code[:500])
