"""
Basic JAX JIT compilation example
Demonstrates @jax.jit decorator and compilation
"""

import jax
import jax.numpy as jnp


@jax.jit
def add_vectors(x, y):
    """Simple vector addition with JIT compilation."""
    return x + y


@jax.jit
def dot_product(x, y):
    """Dot product with JIT compilation."""
    return jnp.dot(x, y)


@jax.jit
def saxpy(a, x, y):
    """SAXPY operation: a*x + y"""
    return a * x + y


@jax.jit
def sum_squares(x):
    """Sum of squares."""
    return jnp.sum(x ** 2)


def main():
    print("=" * 60)
    print("JAX Basic JIT Examples")
    print("=" * 60)
    
    # Test add_vectors
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    result = add_vectors(x, y)
    print(f"\nadd_vectors({x}, {y}) = {result}")
    
    # Test dot_product
    result = dot_product(x, y)
    print(f"dot_product({x}, {y}) = {result}")
    
    # Test saxpy
    a = 2.0
    result = saxpy(a, x, y)
    print(f"saxpy({a}, {x}, {y}) = {result}")
    
    # Test sum_squares
    result = sum_squares(x)
    print(f"sum_squares({x}) = {result}")
    
    print("\n" + "=" * 60)
    print("All examples ran successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
