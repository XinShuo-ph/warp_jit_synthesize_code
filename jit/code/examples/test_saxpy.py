"""Test SAXPY (Scalar A*X Plus Y) with JAX."""
import jax
import jax.numpy as jnp


def saxpy_kernel(alpha, x, y):
    """Compute alpha * x + y."""
    return alpha * x + y


if __name__ == "__main__":
    n = 10
    alpha = 2.5
    x = jnp.array([float(i) for i in range(n)], dtype=jnp.float32)
    y = jnp.array([float(i) for i in range(n)], dtype=jnp.float32)
    
    # JIT compile
    jitted_saxpy = jax.jit(saxpy_kernel)
    
    result = jitted_saxpy(alpha, x, y)
    print("Result:", result)
    print("Expected:", [alpha * i + i for i in range(n)])
    print("Kernel compiled and executed successfully!")
