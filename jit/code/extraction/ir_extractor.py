"""IR Extractor - Extracts HLO and LLVM IR from JAX JIT-compiled functions."""
from dataclasses import dataclass
from typing import Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax._src.lib import xla_client


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    function_name: str
    python_source: str
    hlo_text: str  # HLO IR (high-level XLA intermediate representation)
    llvm_ir: Optional[str] = None  # LLVM IR (low-level, compiled code)
    optimized_hlo: Optional[str] = None  # Optimized HLO after XLA passes


def extract_ir(func: Callable, *args, enable_backward: bool = True, **kwargs) -> ExtractedIR:
    """
    Extract IR (HLO and LLVM) from a JAX function.
    
    Args:
        func: A JAX-compatible function (can be decorated with @jax.jit)
        *args: Sample arguments to trace the function
        enable_backward: Whether to include backward pass (gradient computation)
        **kwargs: Additional keyword arguments for the function
        
    Returns:
        ExtractedIR containing Python source and generated HLO/LLVM IR
    """
    # Get the function name
    func_name = func.__name__ if hasattr(func, '__name__') else 'jax_function'
    
    # Get Python source
    import inspect
    try:
        python_source = inspect.getsource(func)
    except (OSError, TypeError):
        python_source = f"# Source not available for {func_name}"
    
    # Compile the function
    if not isinstance(func, jax.stages.Wrapped):
        compiled_func = jax.jit(func)
    else:
        compiled_func = func
    
    # Lower to get computation
    lowered = jax.jit(func).lower(*args, **kwargs)
    
    # Get HLO text (unoptimized)
    hlo_text = lowered.as_text()
    
    # Get optimized HLO
    try:
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
    except Exception:
        optimized_hlo = None
    
    # Try to get LLVM IR
    llvm_ir = None
    try:
        # Compile and get LLVM IR if available
        compiled = lowered.compile()
        # Try to extract LLVM IR from compilation
        if hasattr(compiled, '_executable'):
            executable = compiled._executable
            if hasattr(executable, 'hlo_modules'):
                # LLVM IR extraction is platform-specific and may not always be available
                llvm_ir = "# LLVM IR extraction not fully supported in current JAX API"
    except Exception:
        pass
    
    # If backward pass is enabled, also extract gradient computation
    if enable_backward:
        try:
            # Create a function that computes value and gradient
            def func_with_grad(*args_inner, **kwargs_inner):
                # Create a scalar loss function for gradient computation
                def scalar_func(*a):
                    result = func(*a, **kwargs_inner)
                    # If result is already scalar, use it; otherwise sum it
                    if hasattr(result, 'shape') and result.shape == ():
                        return result
                    return jnp.sum(result)
                
                value = func(*args_inner, **kwargs_inner)
                grad = jax.grad(scalar_func)(*args_inner)
                return value, grad
            
            # Lower the gradient function
            lowered_grad = jax.jit(func_with_grad).lower(*args, **kwargs)
            grad_hlo = lowered_grad.as_text()
            
            # Append gradient HLO to main HLO
            hlo_text += "\n\n# ===== BACKWARD PASS (GRADIENT) =====\n\n" + grad_hlo
            
            # Also get optimized gradient HLO
            try:
                compiled_grad = lowered_grad.compile()
                grad_optimized = compiled_grad.as_text()
                if optimized_hlo:
                    optimized_hlo += "\n\n# ===== BACKWARD PASS (GRADIENT) =====\n\n" + grad_optimized
            except Exception:
                pass
                
        except Exception as e:
            # Some functions may not be differentiable
            hlo_text += f"\n\n# Note: Gradient computation not available ({str(e)})"
    
    return ExtractedIR(
        function_name=func_name,
        python_source=python_source,
        hlo_text=hlo_text,
        llvm_ir=llvm_ir,
        optimized_hlo=optimized_hlo,
    )


def extract_ir_pair(func: Callable, *args, backend: str = "cpu", **kwargs) -> tuple[str, str]:
    """
    Extract Python→HLO pair suitable for LLM training.
    
    Args:
        func: A JAX function
        *args: Sample arguments for tracing
        backend: "cpu" or "gpu" (affects compilation target)
        **kwargs: Additional keyword arguments
        
    Returns:
        Tuple of (python_source, hlo_code)
    """
    # Set backend
    original_backend = jax.default_backend()
    try:
        if backend == "gpu":
            jax.config.update('jax_platform_name', 'gpu')
        else:
            jax.config.update('jax_platform_name', 'cpu')
    except Exception:
        pass
    
    ir = extract_ir(func, *args, **kwargs)
    
    # Restore original backend
    try:
        jax.config.update('jax_platform_name', original_backend)
    except Exception:
        pass
    
    # Use optimized HLO if available, otherwise use unoptimized
    hlo_code = ir.optimized_hlo if ir.optimized_hlo else ir.hlo_text
    
    return (ir.python_source, hlo_code)


if __name__ == "__main__":
    # Test with a simple function
    print("=== Testing JAX IR Extractor ===\n")
    
    @jax.jit
    def test_function(a, b):
        """Simple test function."""
        return a * 2.0 + b
    
    # Create sample inputs
    a = jnp.array([1.0, 2.0, 3.0])
    b = jnp.array([4.0, 5.0, 6.0])
    
    ir = extract_ir(test_function, a, b)
    
    print("=== Function Name ===")
    print(ir.function_name)
    
    print("\n=== Python Source ===")
    print(ir.python_source)
    
    print("\n=== HLO IR (first 1500 chars) ===")
    print(ir.hlo_text[:1500])
    
    print("\n=== Optimized HLO available ===")
    print("Yes" if ir.optimized_hlo else "No")
    
    if ir.optimized_hlo:
        print("\n=== Optimized HLO (first 1500 chars) ===")
        print(ir.optimized_hlo[:1500])
    
    print("\n=== LLVM IR available ===")
    print("Yes" if ir.llvm_ir else "No")
