"""IR extraction utilities for JAX functions.

Extracts JAXPR and XLA HLO representations from Python functions.
"""
import inspect
import json
import textwrap
from typing import Callable, Any

import jax
import jax.numpy as jnp


def extract_source(fn: Callable) -> str:
    """Extract the source code of a function."""
    try:
        source = inspect.getsource(fn)
        return textwrap.dedent(source)
    except (OSError, TypeError):
        return ""


def extract_jaxpr(fn: Callable, *example_args) -> str:
    """Extract JAXPR representation from a function.
    
    Args:
        fn: The function to trace
        *example_args: Example arguments for tracing
    
    Returns:
        String representation of JAXPR
    """
    closed_jaxpr = jax.make_jaxpr(fn)(*example_args)
    return str(closed_jaxpr)


def extract_hlo(fn: Callable, *example_args) -> str:
    """Extract XLA HLO (StableHLO) representation from a function.
    
    Args:
        fn: The function to compile
        *example_args: Example arguments for compilation
    
    Returns:
        String representation of XLA HLO module
    """
    jitted = jax.jit(fn)
    lowered = jitted.lower(*example_args)
    return lowered.as_text()


def extract_all(fn: Callable, *example_args) -> dict:
    """Extract all IR representations from a function.
    
    Args:
        fn: The function to analyze
        *example_args: Example arguments for tracing/compilation
    
    Returns:
        Dictionary containing:
        - name: Function name
        - source: Python source code
        - jaxpr: JAXPR representation
        - hlo: XLA HLO representation
        - input_shapes: Shapes of example inputs
    """
    input_shapes = []
    for arg in example_args:
        if hasattr(arg, 'shape'):
            input_shapes.append(list(arg.shape))
        else:
            input_shapes.append(None)
    
    return {
        "name": fn.__name__,
        "source": extract_source(fn),
        "jaxpr": extract_jaxpr(fn, *example_args),
        "hlo": extract_hlo(fn, *example_args),
        "input_shapes": input_shapes,
    }


def save_pair(data: dict, filepath: str) -> None:
    """Save a Python→IR pair to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_pair(filepath: str) -> dict:
    """Load a Python→IR pair from a JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


if __name__ == "__main__":
    # Test the extraction functions
    def simple_add(x, y):
        return x + y
    
    def saxpy(a, x, y):
        return a * x + y
    
    def softmax(x):
        exp_x = jnp.exp(x - jnp.max(x))
        return exp_x / jnp.sum(exp_x)
    
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.1, 0.2, 0.3, 0.4])
    
    print("=== Test simple_add ===")
    result = extract_all(simple_add, x, y)
    print(f"Name: {result['name']}")
    print(f"Source:\n{result['source']}")
    print(f"JAXPR: {result['jaxpr']}")
    print(f"Input shapes: {result['input_shapes']}")
    
    print("\n=== Test softmax ===")
    result = extract_all(softmax, x)
    print(f"Name: {result['name']}")
    print(f"JAXPR: {result['jaxpr']}")
