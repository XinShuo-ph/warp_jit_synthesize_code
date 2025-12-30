"""JAX IR Extractor - Extracts generated HLO/XLA code from JAX functions.

JAX compiles functions to XLA's HLO (High Level Optimizer) intermediate representation.
This module extracts the HLO text representation which can be used for training data.
"""
from dataclasses import dataclass
from typing import Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax._src.stages import Lowered


@dataclass
class ExtractedIR:
    """Container for extracted IR from a JAX function."""
    function_name: str
    python_source: str
    hlo_text: str  # HLO text representation
    hlo_optimized: Optional[str] = None  # Optimized HLO
    mlir_text: Optional[str] = None  # MLIR/StableHLO representation


def get_sample_inputs(func: Callable, seed: int = 42) -> tuple:
    """
    Generate sample inputs for a JAX function based on its signature.
    
    This is a heuristic approach - we try to infer input shapes from
    the function source or use reasonable defaults.
    """
    import inspect
    import re
    
    key = jax.random.PRNGKey(seed)
    sig = inspect.signature(func)
    inputs = []
    
    for param_name, param in sig.parameters.items():
        annotation = param.annotation
        
        # Skip self parameter
        if param_name == 'self':
            continue
            
        # Handle different parameter types based on naming conventions
        if param_name in ('n', 'num', 'count', 'size'):
            inputs.append(5)
        elif param_name in ('alpha', 'beta', 'scale', 'gamma'):
            inputs.append(2.0)
        elif param_name == 'key':
            inputs.append(key)
        else:
            # Default to a small array
            key, subkey = jax.random.split(key)
            # Use different shapes for different parameter positions
            if 'vec' in param_name or param_name in ('a', 'b', 'c') and len(inputs) < 2:
                # For vector operations, use (batch, 3) shape
                inputs.append(jax.random.normal(subkey, (8, 3)))
            elif 'mat' in param_name:
                # For matrix operations
                inputs.append(jax.random.normal(subkey, (8, 8)))
            else:
                # Default 1D array
                inputs.append(jax.random.normal(subkey, (16,)))
    
    return tuple(inputs)


def extract_ir(func: Callable, sample_inputs: tuple = None, 
               enable_optimized: bool = True) -> ExtractedIR:
    """
    Extract IR (HLO/MLIR) from a JAX function.
    
    Args:
        func: A JAX function (can be jit-compiled or not)
        sample_inputs: Sample inputs to trace the function. If None, will attempt
                      to generate sample inputs automatically.
        enable_optimized: Whether to also extract optimized HLO
        
    Returns:
        ExtractedIR containing Python source and generated HLO/MLIR code
    """
    import inspect
    
    # Get function name
    func_name = getattr(func, '__name__', 'anonymous')
    
    # Get Python source
    try:
        python_source = inspect.getsource(func)
    except (TypeError, OSError):
        python_source = f"# Source not available for {func_name}"
    
    # Generate sample inputs if not provided
    if sample_inputs is None:
        sample_inputs = get_sample_inputs(func)
    
    # Ensure function is JIT compiled
    jitted_func = jax.jit(func) if not hasattr(func, 'lower') else func
    
    # Lower the function to get HLO
    lowered = jitted_func.lower(*sample_inputs)
    
    # Extract HLO text representation
    hlo_text = lowered.as_text()
    
    # Extract optimized HLO if requested
    hlo_optimized = None
    if enable_optimized:
        try:
            compiled = lowered.compile()
            hlo_optimized = compiled.as_text()
        except Exception:
            pass
    
    # Extract MLIR/StableHLO representation
    mlir_text = None
    try:
        # Try to get MLIR representation
        mlir_text = lowered.as_text(dialect="stablehlo")
    except Exception:
        try:
            # Fallback to MHLO
            mlir_text = lowered.as_text(dialect="mhlo")
        except Exception:
            pass
    
    return ExtractedIR(
        function_name=func_name,
        python_source=python_source,
        hlo_text=hlo_text,
        hlo_optimized=hlo_optimized,
        mlir_text=mlir_text,
    )


def extract_ir_with_grad(func: Callable, sample_inputs: tuple = None) -> tuple[ExtractedIR, ExtractedIR]:
    """
    Extract IR for both forward and gradient functions.
    
    Args:
        func: A JAX function that returns a scalar (for gradient computation)
        sample_inputs: Sample inputs to trace the function
        
    Returns:
        Tuple of (forward_ir, gradient_ir)
    """
    import inspect
    
    # Generate sample inputs if not provided
    if sample_inputs is None:
        sample_inputs = get_sample_inputs(func)
    
    # Extract forward IR
    forward_ir = extract_ir(func, sample_inputs)
    
    # Create gradient function
    # We compute gradient with respect to the first argument
    grad_func = jax.grad(lambda *args: jnp.sum(func(*args)))
    
    # Extract gradient IR
    grad_ir = extract_ir(grad_func, sample_inputs)
    grad_ir.function_name = f"{forward_ir.function_name}_grad"
    
    return forward_ir, grad_ir


def extract_ir_pair(func: Callable, sample_inputs: tuple = None) -> tuple[str, str]:
    """
    Extract Python→HLO pair suitable for LLM training.
    
    Args:
        func: A JAX function
        sample_inputs: Sample inputs
        
    Returns:
        Tuple of (python_source, hlo_code)
    """
    ir = extract_ir(func, sample_inputs)
    return (ir.python_source, ir.hlo_text)


def extract_combined_ir(func: Callable, sample_inputs: tuple = None) -> str:
    """
    Extract combined forward + backward HLO for a function.
    
    This creates a combined representation similar to what Warp generates
    with both forward and backward (adjoint) code.
    """
    forward_ir, grad_ir = extract_ir_with_grad(func, sample_inputs)
    
    combined = f"""// Forward Function: {forward_ir.function_name}
{forward_ir.hlo_text}

// Gradient Function: {grad_ir.function_name}
{grad_ir.hlo_text}
"""
    return combined


if __name__ == "__main__":
    # Test with a simple function
    print("=== JAX IR Extraction Test ===\n")
    
    @jax.jit
    def test_function(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
        """Simple test function."""
        return a * 2.0 + b
    
    # Generate sample inputs
    key = jax.random.PRNGKey(0)
    a = jax.random.normal(key, (8,))
    b = jax.random.normal(jax.random.PRNGKey(1), (8,))
    
    ir = extract_ir(test_function, (a, b))
    
    print("=== Function Name ===")
    print(ir.function_name)
    
    print("\n=== Python Source ===")
    print(ir.python_source)
    
    print("\n=== HLO Text (first 2000 chars) ===")
    print(ir.hlo_text[:2000] if len(ir.hlo_text) > 2000 else ir.hlo_text)
    
    print("\n=== MLIR Available ===")
    print("Yes" if ir.mlir_text else "No")
    if ir.mlir_text:
        print("\n=== MLIR Text (first 1500 chars) ===")
        print(ir.mlir_text[:1500] if len(ir.mlir_text) > 1500 else ir.mlir_text)
    
    # Test gradient extraction
    print("\n\n=== Testing Gradient Extraction ===")
    
    def scalar_func(x: jnp.ndarray) -> float:
        """Function that returns a scalar."""
        return jnp.sum(x ** 2)
    
    forward_ir, grad_ir = extract_ir_with_grad(scalar_func, (jax.random.normal(key, (8,)),))
    
    print(f"\nForward function: {forward_ir.function_name}")
    print(f"Gradient function: {grad_ir.function_name}")
    print(f"\nGradient HLO (first 1000 chars):")
    print(grad_ir.hlo_text[:1000] if len(grad_ir.hlo_text) > 1000 else grad_ir.hlo_text)
