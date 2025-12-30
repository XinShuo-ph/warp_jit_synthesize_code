"""IR Extractor - Extracts generated XLA HLO/MHLO code from JAX functions."""
from dataclasses import dataclass
from typing import Optional, Callable
import jax
import jax.numpy as jnp
from jax import make_jaxpr
from jax._src.compiler import get_compile_options
from jax.stages import Compiled


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    kernel_name: str
    python_source: str
    hlo_text: str  # HLO text representation
    optimized_hlo: Optional[str] = None  # Optimized HLO after compiler passes
    mhlo_text: Optional[str] = None  # MHLO (MLIR HLO) representation


def extract_ir(func: Callable, example_inputs: tuple, kernel_name: str = None) -> ExtractedIR:
    """
    Extract IR (HLO/MHLO code) from a JAX function.
    
    Args:
        func: A JAX-compatible function (typically decorated with @jax.jit)
        example_inputs: Example inputs to trace the function
        kernel_name: Optional name for the kernel
        
    Returns:
        ExtractedIR containing Python source and generated HLO/MHLO code
    """
    if kernel_name is None:
        kernel_name = func.__name__ if hasattr(func, '__name__') else "jax_kernel"
    
    # Get Python source
    import inspect
    try:
        python_source = inspect.getsource(func)
    except (OSError, TypeError):
        python_source = f"# Source not available for {kernel_name}"
    
    # Create a jitted version if not already
    if not hasattr(func, 'lower'):
        jitted_func = jax.jit(func)
    else:
        jitted_func = func
    
    # Lower the function to get HLO
    lowered = jitted_func.lower(*example_inputs)
    
    # Get the HLO text representation
    hlo_text = lowered.as_text()
    
    # Get the compiler IR (optimized HLO after passes)
    try:
        compiled = lowered.compile()
        # Get the optimized HLO
        optimized_hlo = compiled.as_text()
    except Exception:
        optimized_hlo = None
    
    # Try to get MHLO representation (MLIR-based)
    mhlo_text = None
    try:
        # MHLO is the MLIR representation before XLA HLO
        mhlo_module = lowered.compiler_ir(dialect='mhlo')
        mhlo_text = str(mhlo_module)
    except Exception:
        pass
    
    return ExtractedIR(
        kernel_name=kernel_name,
        python_source=python_source,
        hlo_text=hlo_text,
        optimized_hlo=optimized_hlo,
        mhlo_text=mhlo_text,
    )


def extract_ir_with_grad(func: Callable, example_inputs: tuple, kernel_name: str = None) -> ExtractedIR:
    """
    Extract IR including gradient computation (backward pass).
    
    Args:
        func: A JAX-compatible function
        example_inputs: Example inputs to trace the function
        kernel_name: Optional name for the kernel
        
    Returns:
        ExtractedIR containing forward and backward pass HLO
    """
    if kernel_name is None:
        kernel_name = func.__name__ if hasattr(func, '__name__') else "jax_kernel"
    
    # Get Python source
    import inspect
    try:
        python_source = inspect.getsource(func)
    except (OSError, TypeError):
        python_source = f"# Source not available for {kernel_name}"
    
    # Create a function that includes gradient computation
    # For array operations, we compute the gradient with respect to the first differentiable arg
    def forward_and_backward(*args):
        # Forward pass
        out = func(*args)
        # For gradient computation, we need a scalar output
        # Sum the output if it's an array
        if hasattr(out, 'shape') and out.shape != ():
            scalar_out = jnp.sum(out)
        else:
            scalar_out = out
        return scalar_out
    
    # Create gradient function
    grad_func = jax.grad(forward_and_backward, argnums=0)
    
    # Combine forward and backward
    def combined_func(*args):
        fwd = forward_and_backward(*args)
        bwd = grad_func(*args)
        return fwd, bwd
    
    # Jit and lower the combined function
    jitted_func = jax.jit(combined_func)
    lowered = jitted_func.lower(*example_inputs)
    
    # Get HLO representations
    hlo_text = lowered.as_text()
    
    try:
        compiled = lowered.compile()
        optimized_hlo = compiled.as_text()
    except Exception:
        optimized_hlo = None
    
    mhlo_text = None
    try:
        mhlo_module = lowered.compiler_ir(dialect='mhlo')
        mhlo_text = str(mhlo_module)
    except Exception:
        pass
    
    return ExtractedIR(
        kernel_name=kernel_name,
        python_source=python_source,
        hlo_text=hlo_text,
        optimized_hlo=optimized_hlo,
        mhlo_text=mhlo_text,
    )


def extract_ir_pair(func: Callable, example_inputs: tuple, 
                   kernel_name: str = None, include_grad: bool = True) -> tuple[str, str]:
    """
    Extract Python→HLO pair suitable for LLM training.
    
    Args:
        func: A JAX function
        example_inputs: Example inputs for tracing
        kernel_name: Optional kernel name
        include_grad: Whether to include gradient computation
        
    Returns:
        Tuple of (python_source, hlo_code)
    """
    if include_grad:
        ir = extract_ir_with_grad(func, example_inputs, kernel_name)
    else:
        ir = extract_ir(func, example_inputs, kernel_name)
    
    # Return python source and HLO (prefer optimized if available)
    hlo_code = ir.optimized_hlo if ir.optimized_hlo else ir.hlo_text
    return (ir.python_source, hlo_code)


if __name__ == "__main__":
    # Test with a simple kernel
    print("=== Testing JAX IR Extractor ===\n")
    
    def test_kernel(a, b):
        """Simple elementwise multiplication kernel."""
        return a * 2.0 + b
    
    # Create example inputs
    a = jnp.array([1.0, 2.0, 3.0, 4.0])
    b = jnp.array([0.5, 1.5, 2.5, 3.5])
    
    ir = extract_ir(test_kernel, (a, b), "test_kernel")
    
    print("=== Kernel Name ===")
    print(ir.kernel_name)
    print("\n=== Python Source ===")
    print(ir.python_source)
    print("\n=== HLO Text (first 1500 chars) ===")
    print(ir.hlo_text[:1500])
    if ir.optimized_hlo:
        print("\n=== Optimized HLO available ===")
        print("Yes")
        print("\n=== Optimized HLO (first 1500 chars) ===")
        print(ir.optimized_hlo[:1500])
    if ir.mhlo_text:
        print("\n=== MHLO available ===")
        print("Yes")
    
    # Test with gradient
    print("\n\n=== Testing with Gradient ===\n")
    ir_grad = extract_ir_with_grad(test_kernel, (a, b), "test_kernel_grad")
    print("=== HLO with Gradient (first 1500 chars) ===")
    print(ir_grad.hlo_text[:1500])
