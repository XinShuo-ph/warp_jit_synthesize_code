"""Basic JAX JIT examples demonstrating IR extraction."""

import jax
import jax.numpy as jnp


def example1_simple_math():
    """Simple math function."""
    def f(x, y):
        return x * y + jnp.sin(x)
    
    # Get jaxpr
    jaxpr = jax.make_jaxpr(f)(1.0, 2.0)
    print("=== Example 1: Simple Math ===")
    print("Python: def f(x, y): return x * y + jnp.sin(x)")
    print("\nJaxpr:")
    print(jaxpr)
    
    # Get HLO
    lowered = jax.jit(f).lower(1.0, 2.0)
    hlo_text = lowered.as_text()
    print("\nHLO (first 500 chars):")
    print(hlo_text[:500])
    print("...")
    return jaxpr, hlo_text


def example2_array_ops():
    """Array operations."""
    def matmul_add(a, b, c):
        return jnp.dot(a, b) + c
    
    a = jnp.ones((3, 4))
    b = jnp.ones((4, 5))
    c = jnp.ones((3, 5))
    
    jaxpr = jax.make_jaxpr(matmul_add)(a, b, c)
    print("\n=== Example 2: Matrix Operations ===")
    print("Python: def matmul_add(a, b, c): return jnp.dot(a, b) + c")
    print("\nJaxpr:")
    print(jaxpr)
    
    lowered = jax.jit(matmul_add).lower(a, b, c)
    hlo_text = lowered.as_text()
    print("\nHLO (first 500 chars):")
    print(hlo_text[:500])
    print("...")
    return jaxpr, hlo_text


def example3_control_flow():
    """Control flow with lax.cond."""
    def relu(x):
        return jax.lax.cond(x > 0, lambda: x, lambda: 0.0)
    
    jaxpr = jax.make_jaxpr(relu)(1.0)
    print("\n=== Example 3: Control Flow (relu) ===")
    print("Python: def relu(x): return jax.lax.cond(x > 0, lambda: x, lambda: 0.0)")
    print("\nJaxpr:")
    print(jaxpr)
    
    lowered = jax.jit(relu).lower(1.0)
    hlo_text = lowered.as_text()
    print("\nHLO (first 500 chars):")
    print(hlo_text[:500])
    print("...")
    return jaxpr, hlo_text


if __name__ == "__main__":
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}\n")
    
    example1_simple_math()
    example2_array_ops()
    example3_control_flow()
    
    print("\n=== All examples completed successfully ===")
