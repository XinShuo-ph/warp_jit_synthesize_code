"""Basic JAX JIT examples demonstrating compilation."""
import jax
import jax.numpy as jnp


@jax.jit
def add(x, y):
    """Element-wise addition."""
    return x + y


@jax.jit
def compute_trig(x):
    """Trigonometric identity: sin^2(x) + cos^2(x) = 1."""
    return jnp.sin(x) ** 2 + jnp.cos(x) ** 2


@jax.jit
def dot_product(a, b):
    """Vector dot product."""
    return jnp.sum(a * b)


@jax.jit
def matmul(A, B):
    """Matrix multiplication."""
    return jnp.dot(A, B)


if __name__ == "__main__":
    # Test examples
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.array([4.0, 5.0, 6.0])
    
    print(f"add: {add(x, y)}")
    print(f"compute_trig: {compute_trig(x)}")
    print(f"dot_product: {dot_product(x, y)}")
    
    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    B = jnp.array([[5.0, 6.0], [7.0, 8.0]])
    print(f"matmul:\n{matmul(A, B)}")
