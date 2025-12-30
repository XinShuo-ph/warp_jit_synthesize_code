#!/usr/bin/env python3
"""
Example 2: Array Operations with JAX
Demonstrates various array operations that benefit from JIT compilation
"""
import jax
import jax.numpy as jnp
from jax import jit


@jit
def vector_operations(x, y):
    """Basic vector operations"""
    add = x + y
    mul = x * y
    dot = jnp.dot(x, y)
    return add, mul, dot


@jit
def matrix_operations(A, B):
    """Matrix operations"""
    matmul = jnp.matmul(A, B)
    transpose = A.T
    trace = jnp.trace(A)
    return matmul, transpose, trace


@jit
def reduction_operations(x):
    """Reduction operations"""
    sum_val = jnp.sum(x)
    mean_val = jnp.mean(x)
    max_val = jnp.max(x)
    min_val = jnp.min(x)
    return sum_val, mean_val, max_val, min_val


@jit
def broadcasting_example(x, y):
    """Broadcasting operations"""
    # x: (3, 1), y: (1, 4) -> broadcasts to (3, 4)
    result = x + y
    return result


def main():
    print("=" * 60)
    print("JAX Array Operations Demo")
    print("=" * 60)
    
    # Vector operations
    print("\n1. Vector Operations")
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    add, mul, dot = vector_operations(x, y)
    print(f"   x = {x}")
    print(f"   y = {y}")
    print(f"   x + y = {add}")
    print(f"   x * y = {mul}")
    print(f"   dot(x, y) = {dot}")
    
    # Matrix operations
    print("\n2. Matrix Operations")
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    matmul, transpose, trace = matrix_operations(A, B)
    print(f"   A = \n{A}")
    print(f"   B = \n{B}")
    print(f"   A @ B = \n{matmul}")
    print(f"   A.T = \n{transpose}")
    print(f"   trace(A) = {trace}")
    
    # Reduction operations
    print("\n3. Reduction Operations")
    x = jnp.arange(10, dtype=jnp.float32)
    sum_val, mean_val, max_val, min_val = reduction_operations(x)
    print(f"   x = {x}")
    print(f"   sum(x) = {sum_val}")
    print(f"   mean(x) = {mean_val}")
    print(f"   max(x) = {max_val}")
    print(f"   min(x) = {min_val}")
    
    # Broadcasting
    print("\n4. Broadcasting")
    x = jnp.array([[1.0], [2.0], [3.0]])  # (3, 1)
    y = jnp.array([[10.0, 20.0, 30.0, 40.0]])  # (1, 4)
    result = broadcasting_example(x, y)
    print(f"   x shape: {x.shape}, y shape: {y.shape}")
    print(f"   x + y (broadcasted) = \n{result}")
    print(f"   result shape: {result.shape}")
    
    print("\n" + "=" * 60)
    print("SUCCESS: All array operations completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
