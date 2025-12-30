"""IR Extractor - Extracts generated HLO/XLA code from JAX functions."""
from dataclasses import dataclass
from typing import Optional, Callable, Tuple, Any
import jax
import jax.numpy as jnp
from jax import grad, jit


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    kernel_name: str
    python_source: str
    hlo_text: str  # HLO (High Level Optimizer) text representation
    jaxpr_text: str  # JAX's intermediate representation
    optimized_hlo: Optional[str] = None  # Optimized HLO after compilation


def get_function_source(func: Callable) -> str:
    """Get the source code of a function."""
    import inspect
    try:
        return inspect.getsource(func)
    except (OSError, TypeError):
        return f"# Source not available for {func.__name__}"


def extract_ir(
    func: Callable,
    sample_args: Tuple[Any, ...],
    enable_backward: bool = True,
    func_name: Optional[str] = None
) -> ExtractedIR:
    """
    Extract IR (HLO/Jaxpr) from a JAX function.
    
    Args:
        func: A JAX-compatible function
        sample_args: Sample input arguments for tracing the function
        enable_backward: Whether to include backward (gradient) code
        func_name: Optional name override for the function
        
    Returns:
        ExtractedIR containing Python source and generated HLO/Jaxpr code
    """
    kernel_name = func_name or getattr(func, '__name__', 'anonymous')
    python_source = get_function_source(func)
    
    # Get Jaxpr (JAX's intermediate representation)
    try:
        jaxpr = jax.make_jaxpr(func)(*sample_args)
        jaxpr_text = str(jaxpr)
    except Exception as e:
        jaxpr_text = f"# Jaxpr extraction failed: {e}"
    
    # Get HLO text (lowered representation before optimization)
    try:
        lowered = jax.jit(func).lower(*sample_args)
        hlo_text = lowered.as_text()
    except Exception as e:
        hlo_text = f"# HLO extraction failed: {e}"
    
    # Get optimized HLO (after XLA compilation)
    optimized_hlo = None
    try:
        lowered = jax.jit(func).lower(*sample_args)
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
    except Exception:
        pass  # Optimized HLO may not always be available
    
    # If backward is enabled, also extract gradient function IR
    if enable_backward:
        try:
            # Create a scalar loss function for gradient extraction
            def scalar_loss(*args):
                result = func(*args)
                if isinstance(result, jnp.ndarray):
                    return jnp.sum(result)
                return result
            
            # Get gradient function
            grad_fn = grad(scalar_loss)
            
            # Get gradient Jaxpr
            grad_jaxpr = jax.make_jaxpr(grad_fn)(*sample_args)
            jaxpr_text += "\n\n# === BACKWARD (GRADIENT) JAXPR ===\n"
            jaxpr_text += str(grad_jaxpr)
            
            # Get gradient HLO
            grad_lowered = jax.jit(grad_fn).lower(*sample_args)
            grad_hlo = grad_lowered.as_text()
            hlo_text += "\n\n// === BACKWARD (GRADIENT) HLO ===\n"
            hlo_text += grad_hlo
            
            # Get optimized gradient HLO
            if optimized_hlo is not None:
                try:
                    grad_compiled = grad_lowered.compile()
                    grad_opt_hlo = grad_compiled.as_text()
                    optimized_hlo += "\n\n// === BACKWARD (GRADIENT) OPTIMIZED HLO ===\n"
                    optimized_hlo += grad_opt_hlo
                except Exception:
                    pass
                    
        except Exception as e:
            jaxpr_text += f"\n\n# Backward extraction failed: {e}"
    
    return ExtractedIR(
        kernel_name=kernel_name,
        python_source=python_source,
        hlo_text=hlo_text,
        jaxpr_text=jaxpr_text,
        optimized_hlo=optimized_hlo,
    )


def extract_ir_pair(func: Callable, sample_args: Tuple[Any, ...], ir_type: str = "hlo") -> Tuple[str, str]:
    """
    Extract Python→IR pair suitable for LLM training.
    
    Args:
        func: A JAX function
        sample_args: Sample input arguments
        ir_type: "hlo", "jaxpr", or "optimized"
        
    Returns:
        Tuple of (python_source, ir_code)
    """
    ir = extract_ir(func, sample_args)
    
    if ir_type == "jaxpr":
        return (ir.python_source, ir.jaxpr_text)
    elif ir_type == "optimized" and ir.optimized_hlo:
        return (ir.python_source, ir.optimized_hlo)
    return (ir.python_source, ir.hlo_text)


if __name__ == "__main__":
    # Test with a simple function
    print("Testing JAX IR Extractor...")
    
    def test_kernel(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        return a * 2.0 + b
    
    # Create sample inputs
    key = jax.random.PRNGKey(42)
    a = jax.random.normal(key, (10,))
    b = jax.random.normal(key, (10,))
    
    ir = extract_ir(test_kernel, (a, b))
    
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== Jaxpr (first 1500 chars) ===")
    print(ir.jaxpr_text[:1500] if len(ir.jaxpr_text) > 1500 else ir.jaxpr_text)
    print("\n=== HLO Text (first 1500 chars) ===")
    print(ir.hlo_text[:1500] if len(ir.hlo_text) > 1500 else ir.hlo_text)
    print("\n=== Optimized HLO available ===")
    print("Yes" if ir.optimized_hlo else "No")
    if ir.optimized_hlo:
        print("\n=== Optimized HLO (first 1500 chars) ===")
        print(ir.optimized_hlo[:1500] if len(ir.optimized_hlo) > 1500 else ir.optimized_hlo)
