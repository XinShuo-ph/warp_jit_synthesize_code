"""IR Extractor - Extracts generated XLA HLO and compiled code from JAX functions."""
from dataclasses import dataclass
from typing import Optional, Callable
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import inspect


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    kernel_name: str
    python_source: str
    hlo_text: str  # XLA HLO representation
    optimized_hlo: Optional[str] = None  # Optimized HLO


def extract_ir(func: Callable, sample_args: tuple, enable_backward: bool = True) -> ExtractedIR:
    """
    Extract IR (XLA HLO code) from a JAX function.
    
    Args:
        func: A JAX-compatible function (can be jitted or not)
        sample_args: Sample arguments to compile the function
        enable_backward: Whether to include backward (gradient) code
        
    Returns:
        ExtractedIR containing Python source and generated XLA HLO code
    """
    # Get the Python source code
    try:
        python_source = inspect.getsource(func)
    except:
        python_source = f"# Source not available for {func.__name__}"
    
    # JIT compile the function to get HLO
    jitted_func = jit(func)
    
    # Lower the function to get HLO representation
    lowered = jax.jit(func).lower(*sample_args)
    
    # Get HLO text representation
    hlo_text = lowered.as_text()
    
    # Get optimized HLO if available
    optimized_hlo = None
    try:
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
    except:
        pass
    
    # If backward pass is enabled, also compile the gradient function
    if enable_backward:
        # Create a scalar output version for grad
        def scalar_output_wrapper(*args):
            result = func(*args)
            # Sum to get scalar for gradient computation
            if isinstance(result, (jnp.ndarray, jax.Array)):
                return jnp.sum(result)
            return result
        
        try:
            grad_func = grad(scalar_output_wrapper)
            grad_lowered = jax.jit(grad_func).lower(*sample_args)
            grad_hlo = grad_lowered.as_text()
            
            # Combine forward and backward HLO
            hlo_text = f"// ===== FORWARD PASS =====\n{hlo_text}\n\n// ===== BACKWARD PASS =====\n{grad_hlo}"
            
            # Also get optimized backward if available
            try:
                grad_compiled = grad_lowered.compile()
                grad_opt_hlo = grad_compiled.as_text()
                if optimized_hlo:
                    optimized_hlo = f"// ===== FORWARD PASS (OPTIMIZED) =====\n{optimized_hlo}\n\n// ===== BACKWARD PASS (OPTIMIZED) =====\n{grad_opt_hlo}"
            except:
                pass
        except Exception as e:
            # If gradient fails, just use forward pass
            hlo_text += f"\n\n// Note: Backward pass generation failed: {e}"
    
    return ExtractedIR(
        kernel_name=func.__name__,
        python_source=python_source,
        hlo_text=hlo_text,
        optimized_hlo=optimized_hlo,
    )


def extract_ir_pair(func: Callable, sample_args: tuple, device: str = "cpu") -> tuple[str, str]:
    """
    Extract Python→HLO pair suitable for LLM training.
    
    Args:
        func: A JAX function
        sample_args: Sample arguments for compilation
        device: "cpu" or "gpu" (JAX handles device-specific compilation)
        
    Returns:
        Tuple of (python_source, hlo_code)
    """
    # Set default device
    with jax.default_device(jax.devices(device if device == "cpu" else "gpu")[0] if device in ["cpu", "gpu"] else jax.devices()[0]):
        ir = extract_ir(func, sample_args)
        # Use optimized HLO if available, otherwise use standard HLO
        code = ir.optimized_hlo if ir.optimized_hlo else ir.hlo_text
        return (ir.python_source, code)


if __name__ == "__main__":
    # Test with a simple kernel
    def test_kernel(a, b):
        """Simple elementwise multiplication kernel."""
        return a * 2.0
    
    # Create sample inputs
    n = 10
    sample_a = jnp.ones(n, dtype=jnp.float32)
    sample_b = jnp.zeros(n, dtype=jnp.float32)
    
    ir = extract_ir(test_kernel, (sample_a, sample_b))
    
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== HLO Code (first 1500 chars) ===")
    print(ir.hlo_text[:1500])
    print("\n=== Optimized HLO available ===")
    print("Yes" if ir.optimized_hlo else "No")
    if ir.optimized_hlo:
        print("\n=== Optimized HLO (first 1500 chars) ===")
        print(ir.optimized_hlo[:1500])
