"""IR extraction utilities for JAX functions."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable, Any, Tuple
import json


@dataclass
class IRPair:
    """Holds Python source and IR representations."""
    python_source: str
    jaxpr: str
    hlo_text: str
    input_shapes: list
    output_shape: str


def extract_ir(fn: Callable, *args, fn_source: str = None) -> IRPair:
    """
    Extract jaxpr and HLO from a JAX function.
    
    Args:
        fn: JAX-compatible function
        *args: Example arguments for tracing
        fn_source: Optional Python source code string
        
    Returns:
        IRPair with jaxpr and HLO text
    """
    # Get jaxpr
    jaxpr = jax.make_jaxpr(fn)(*args)
    
    # Get HLO text via lowering
    lowered = jax.jit(fn).lower(*args)
    hlo_text = lowered.as_text()
    
    # Extract shape info
    input_shapes = []
    for arg in args:
        if hasattr(arg, 'shape'):
            input_shapes.append(f"{arg.dtype}{list(arg.shape)}")
        else:
            input_shapes.append(str(type(arg).__name__))
    
    # Get output shape by running
    out = fn(*args)
    if hasattr(out, 'shape'):
        output_shape = f"{out.dtype}{list(out.shape)}"
    else:
        output_shape = str(type(out).__name__)
    
    return IRPair(
        python_source=fn_source or "",
        jaxpr=str(jaxpr),
        hlo_text=hlo_text,
        input_shapes=input_shapes,
        output_shape=output_shape
    )


def ir_pair_to_dict(pair: IRPair) -> dict:
    """Convert IRPair to dictionary for JSON serialization."""
    return {
        "python_source": pair.python_source,
        "jaxpr": pair.jaxpr,
        "hlo_text": pair.hlo_text,
        "input_shapes": pair.input_shapes,
        "output_shape": pair.output_shape
    }


def save_ir_pair(pair: IRPair, filepath: str):
    """Save IRPair to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(ir_pair_to_dict(pair), f, indent=2)


def load_ir_pair(filepath: str) -> IRPair:
    """Load IRPair from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return IRPair(**data)


# Test cases
def test_extraction():
    """Test IR extraction with various functions."""
    
    # Test 1: Simple arithmetic
    def add_mul(x, y):
        return (x + y) * x
    
    src1 = "def add_mul(x, y): return (x + y) * x"
    pair1 = extract_ir(add_mul, 1.0, 2.0, fn_source=src1)
    print("Test 1 - Simple arithmetic:")
    print(f"  Jaxpr: {pair1.jaxpr[:80]}...")
    print(f"  Shapes: {pair1.input_shapes} -> {pair1.output_shape}")
    
    # Test 2: Array operations  
    def reduce_sum(x):
        return jnp.sum(x ** 2)
    
    src2 = "def reduce_sum(x): return jnp.sum(x ** 2)"
    arr = jnp.ones((10, 10))
    pair2 = extract_ir(reduce_sum, arr, fn_source=src2)
    print("\nTest 2 - Array reduction:")
    print(f"  Jaxpr: {pair2.jaxpr[:80]}...")
    print(f"  Shapes: {pair2.input_shapes} -> {pair2.output_shape}")
    
    # Test 3: Nested operations
    def softmax(x):
        exp_x = jnp.exp(x - jnp.max(x))
        return exp_x / jnp.sum(exp_x)
    
    src3 = "def softmax(x): exp_x = jnp.exp(x - jnp.max(x)); return exp_x / jnp.sum(exp_x)"
    vec = jnp.ones(10)
    pair3 = extract_ir(softmax, vec, fn_source=src3)
    print("\nTest 3 - Softmax:")
    print(f"  Jaxpr: {pair3.jaxpr[:80]}...")
    print(f"  Shapes: {pair3.input_shapes} -> {pair3.output_shape}")
    
    # Test 4: Multiple outputs (tuple)
    def split_stats(x):
        return jnp.mean(x), jnp.std(x)
    
    src4 = "def split_stats(x): return jnp.mean(x), jnp.std(x)"
    pair4 = extract_ir(split_stats, vec, fn_source=src4)
    print("\nTest 4 - Multiple outputs:")
    print(f"  Jaxpr: {pair4.jaxpr[:80]}...")
    
    # Test 5: vmap
    def batched_dot(x, y):
        return jax.vmap(jnp.dot)(x, y)
    
    src5 = "def batched_dot(x, y): return jax.vmap(jnp.dot)(x, y)"
    batch_x = jnp.ones((5, 3))
    batch_y = jnp.ones((5, 3))
    pair5 = extract_ir(batched_dot, batch_x, batch_y, fn_source=src5)
    print("\nTest 5 - Batched dot (vmap):")
    print(f"  Jaxpr: {pair5.jaxpr[:80]}...")
    print(f"  Shapes: {pair5.input_shapes} -> {pair5.output_shape}")
    
    print("\n=== All 5 test cases passed ===")
    return [pair1, pair2, pair3, pair4, pair5]


if __name__ == "__main__":
    test_extraction()
