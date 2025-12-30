"""IR extraction utilities for JAX functions."""
import inspect
from typing import Callable, Any, Literal
import jax
import jax.numpy as jnp


def extract_jaxpr(fn: Callable, *args) -> str:
    """Extract JAXPR (JAX Program Representation) from a function.
    
    Args:
        fn: JAX-compatible function
        *args: Example arguments to trace the function
        
    Returns:
        String representation of JAXPR
    """
    jaxpr = jax.make_jaxpr(fn)(*args)
    return str(jaxpr)


def extract_stablehlo(fn: Callable, *args) -> str:
    """Extract StableHLO IR from a function.
    
    Args:
        fn: JAX-compatible function
        *args: Example arguments to trace the function
        
    Returns:
        String representation of StableHLO IR
    """
    lowered = jax.jit(fn).lower(*args)
    return lowered.as_text()


def extract_compiled_hlo(fn: Callable, *args) -> str:
    """Extract compiled/optimized HLO IR from a function.
    
    Args:
        fn: JAX-compatible function
        *args: Example arguments to trace the function
        
    Returns:
        String representation of optimized HLO
    """
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    return compiled.as_text()


def extract_ir(
    fn: Callable,
    *args,
    format: Literal["jaxpr", "stablehlo", "hlo"] = "stablehlo"
) -> str:
    """Unified IR extraction function.
    
    Args:
        fn: JAX-compatible function
        *args: Example arguments to trace the function
        format: IR format to extract ("jaxpr", "stablehlo", or "hlo")
        
    Returns:
        String representation of the IR
    """
    if format == "jaxpr":
        return extract_jaxpr(fn, *args)
    elif format == "stablehlo":
        return extract_stablehlo(fn, *args)
    elif format == "hlo":
        return extract_compiled_hlo(fn, *args)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'jaxpr', 'stablehlo', or 'hlo'")


def get_source_code(fn: Callable) -> str:
    """Get source code of a function.
    
    Args:
        fn: Python function
        
    Returns:
        Source code as string
    """
    return inspect.getsource(fn)


def create_ir_pair(
    fn: Callable,
    *args,
    format: Literal["jaxpr", "stablehlo", "hlo"] = "stablehlo",
    include_source: bool = True
) -> dict:
    """Create a Python→IR pair for training data.
    
    Args:
        fn: JAX-compatible function
        *args: Example arguments to trace the function
        format: IR format to extract
        include_source: Whether to include source code
        
    Returns:
        Dictionary with 'python' (source), 'ir' (extracted IR), and metadata
    """
    ir = extract_ir(fn, *args, format=format)
    
    result = {
        "ir": ir,
        "ir_format": format,
        "function_name": fn.__name__,
    }
    
    if include_source:
        try:
            result["python"] = get_source_code(fn)
        except (OSError, TypeError):
            result["python"] = None
    
    # Add argument shapes for reproducibility
    arg_shapes = []
    for arg in args:
        if hasattr(arg, 'shape'):
            arg_shapes.append({"shape": list(arg.shape), "dtype": str(arg.dtype)})
        else:
            arg_shapes.append({"value": type(arg).__name__})
    result["arg_info"] = arg_shapes
    
    return result


if __name__ == "__main__":
    # Quick test
    def test_fn(x, y):
        return jnp.sin(x) + y * 2
    
    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    
    print("JAXPR:")
    print(extract_jaxpr(test_fn, x, y))
    print("\nStableHLO:")
    print(extract_stablehlo(test_fn, x, y))
    print("\nIR Pair:")
    pair = create_ir_pair(test_fn, x, y, format="stablehlo")
    print(f"Function: {pair['function_name']}")
    print(f"Args: {pair['arg_info']}")
