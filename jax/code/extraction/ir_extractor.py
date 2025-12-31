"""IR extraction utilities for JAX JIT functions."""
from typing import Callable, Any, NamedTuple
import jax
import jax.numpy as jnp


class IRPair(NamedTuple):
    """Container for Python function and its extracted IRs."""
    python_source: str
    jaxpr: str
    stablehlo: str
    xla_hlo: str


def extract_jaxpr(fn: Callable, *example_args) -> str:
    """Extract JAXPR (JAX Primitive Representation) from a function.
    
    Args:
        fn: The function to extract IR from (does not need @jax.jit)
        *example_args: Example arguments to trace the function
    
    Returns:
        String representation of the JAXPR
    """
    jaxpr = jax.make_jaxpr(fn)(*example_args)
    return str(jaxpr)


def extract_stablehlo(fn: Callable, *example_args) -> str:
    """Extract StableHLO IR from a JIT-compiled function.
    
    Args:
        fn: The function to extract IR from
        *example_args: Example arguments to trace the function
    
    Returns:
        StableHLO text representation
    """
    lowered = jax.jit(fn).lower(*example_args)
    return lowered.as_text()


def extract_xla_hlo(fn: Callable, *example_args) -> str:
    """Extract XLA HLO from a JIT-compiled function.
    
    Args:
        fn: The function to extract IR from
        *example_args: Example arguments to trace the function
    
    Returns:
        XLA HLO text representation
    """
    lowered = jax.jit(fn).lower(*example_args)
    hlo = lowered.compiler_ir(dialect='hlo')
    return hlo.as_hlo_text()


def extract_all_ir(fn: Callable, *example_args, include_source: bool = True) -> IRPair:
    """Extract all IR representations from a function.
    
    Args:
        fn: The function to extract IR from
        *example_args: Example arguments to trace the function
        include_source: Whether to include Python source code
    
    Returns:
        IRPair containing all representations
    """
    import inspect
    
    python_source = ""
    if include_source:
        try:
            python_source = inspect.getsource(fn)
        except (OSError, TypeError):
            python_source = "<source unavailable>"
    
    jaxpr = extract_jaxpr(fn, *example_args)
    stablehlo = extract_stablehlo(fn, *example_args)
    xla_hlo = extract_xla_hlo(fn, *example_args)
    
    return IRPair(
        python_source=python_source,
        jaxpr=jaxpr,
        stablehlo=stablehlo,
        xla_hlo=xla_hlo
    )


def demo():
    """Demonstrate IR extraction on sample functions."""
    
    def relu(x):
        return jnp.maximum(x, 0)
    
    def mlp_layer(x, w1, b1, w2, b2):
        h = jax.nn.relu(jnp.dot(x, w1) + b1)
        return jnp.dot(h, w2) + b2
    
    # Demo 1: Simple ReLU
    print("=" * 60)
    print("DEMO 1: ReLU function")
    print("=" * 60)
    ir = extract_all_ir(relu, jnp.zeros((4, 8)))
    print("\n--- Python Source ---")
    print(ir.python_source)
    print("\n--- JAXPR ---")
    print(ir.jaxpr)
    print("\n--- XLA HLO ---")
    print(ir.xla_hlo)
    
    # Demo 2: MLP layer
    print("\n" + "=" * 60)
    print("DEMO 2: MLP Layer")
    print("=" * 60)
    x = jnp.zeros((4, 8))
    w1, b1 = jnp.zeros((8, 16)), jnp.zeros((16,))
    w2, b2 = jnp.zeros((16, 4)), jnp.zeros((4,))
    ir = extract_all_ir(mlp_layer, x, w1, b1, w2, b2)
    print("\n--- Python Source ---")
    print(ir.python_source)
    print("\n--- XLA HLO ---")
    print(ir.xla_hlo)


if __name__ == "__main__":
    demo()
