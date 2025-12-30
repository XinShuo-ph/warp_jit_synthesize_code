"""IR Extraction utilities for JAX functions."""
import inspect
import textwrap
from dataclasses import dataclass
from typing import Callable, Any, Tuple

import jax
import jax.numpy as jnp
from jax import jit


@dataclass
class IRPair:
    """Holds Python source and generated StableHLO IR for a function."""
    function_name: str
    python_source: str
    stablehlo_ir: str
    cost_analysis: dict


def extract_ir(func: Callable, *args, **kwargs) -> IRPair:
    """
    Extract Python source and StableHLO IR from a JAX function.
    
    Args:
        func: A JAX-compatible function (can be @jit decorated or not)
        *args: Example arguments for lowering the function
        **kwargs: Example keyword arguments for lowering the function
    
    Returns:
        IRPair containing Python source and StableHLO IR
    
    Example:
        >>> def add(x, y):
        ...     return x + y
        >>> x = jnp.array([1., 2., 3.])
        >>> y = jnp.array([4., 5., 6.])
        >>> pair = extract_ir(add, x, y)
    """
    # Get Python source
    try:
        python_source = inspect.getsource(func)
        # Clean up indentation
        python_source = textwrap.dedent(python_source)
    except OSError:
        # If source is not available (e.g., interactive shell), use repr
        python_source = f"# Source not available\ndef {func.__name__}(*args, **kwargs): ..."
    
    # Get function name
    function_name = func.__name__
    
    # Ensure function is jitted for lowering
    if not hasattr(func, 'lower'):
        func = jit(func)
    
    # Lower the function to get IR
    lowered = func.lower(*args, **kwargs)
    
    # Extract StableHLO IR
    stablehlo_module = lowered.compiler_ir(dialect='stablehlo')
    stablehlo_ir = str(stablehlo_module)
    
    # Get cost analysis
    try:
        cost_analysis = lowered.cost_analysis()
    except Exception:
        cost_analysis = {}
    
    return IRPair(
        function_name=function_name,
        python_source=python_source,
        stablehlo_ir=stablehlo_ir,
        cost_analysis=cost_analysis
    )


def extract_ir_from_jitted(jitted_func: Callable, *args, **kwargs) -> IRPair:
    """
    Extract IR from an already-jitted function.
    
    This is useful when you have a pre-decorated @jit function.
    """
    # Get the original function for source extraction
    if hasattr(jitted_func, '__wrapped__'):
        original_func = jitted_func.__wrapped__
        python_source = inspect.getsource(original_func)
        python_source = textwrap.dedent(python_source)
        function_name = original_func.__name__
    else:
        function_name = jitted_func.__name__
        try:
            python_source = inspect.getsource(jitted_func)
            python_source = textwrap.dedent(python_source)
        except OSError:
            python_source = f"# Source not available\ndef {function_name}(*args, **kwargs): ..."
    
    # Lower and extract IR
    lowered = jitted_func.lower(*args, **kwargs)
    stablehlo_module = lowered.compiler_ir(dialect='stablehlo')
    stablehlo_ir = str(stablehlo_module)
    
    try:
        cost_analysis = lowered.cost_analysis()
    except Exception:
        cost_analysis = {}
    
    return IRPair(
        function_name=function_name,
        python_source=python_source,
        stablehlo_ir=stablehlo_ir,
        cost_analysis=cost_analysis
    )


def extract_with_grad(func: Callable, *args, **kwargs) -> Tuple[IRPair, IRPair]:
    """
    Extract IR for both forward and backward (gradient) passes.
    
    Args:
        func: A JAX function that takes arguments and returns a scalar or array
        *args: Example arguments for the function
        **kwargs: Example keyword arguments
    
    Returns:
        Tuple of (forward_ir_pair, backward_ir_pair)
    
    Example:
        >>> def loss_fn(x, y):
        ...     return jnp.sum((x - y) ** 2)
        >>> x = jnp.array([1., 2., 3.])
        >>> y = jnp.array([4., 5., 6.])
        >>> fwd, bwd = extract_with_grad(loss_fn, x, y)
    """
    # Extract forward pass
    forward_pair = extract_ir(func, *args, **kwargs)
    
    # Create gradient function
    grad_func = jax.grad(func)
    
    # Extract backward pass
    # For grad, we only pass the first argument (the one we're differentiating w.r.t.)
    backward_pair = extract_ir(grad_func, *args, **kwargs)
    backward_pair.function_name = f"{func.__name__}_grad"
    
    return forward_pair, backward_pair


def save_ir_pair(pair: IRPair, output_path: str) -> None:
    """
    Save an IR pair to a file in a readable format.
    
    Args:
        pair: The IRPair to save
        output_path: Path to save the file
    """
    with open(output_path, 'w') as f:
        f.write(f"# Function: {pair.function_name}\n")
        f.write(f"# Cost Analysis: {pair.cost_analysis}\n")
        f.write("\n" + "=" * 80 + "\n")
        f.write("# PYTHON SOURCE\n")
        f.write("=" * 80 + "\n\n")
        f.write(pair.python_source)
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("# STABLEHLO IR\n")
        f.write("=" * 80 + "\n\n")
        f.write(pair.stablehlo_ir)
        f.write("\n")


def compare_ir_pairs(pair1: IRPair, pair2: IRPair) -> dict:
    """
    Compare two IR pairs and return statistics.
    
    Returns:
        Dictionary with comparison metrics
    """
    return {
        'name1': pair1.function_name,
        'name2': pair2.function_name,
        'python_lines1': len(pair1.python_source.splitlines()),
        'python_lines2': len(pair2.python_source.splitlines()),
        'ir_lines1': len(pair1.stablehlo_ir.splitlines()),
        'ir_lines2': len(pair2.stablehlo_ir.splitlines()),
        'flops1': pair1.cost_analysis.get('flops', 0),
        'flops2': pair2.cost_analysis.get('flops', 0),
        'bytes_accessed1': pair1.cost_analysis.get('bytes accessed', 0),
        'bytes_accessed2': pair2.cost_analysis.get('bytes accessed', 0),
    }
