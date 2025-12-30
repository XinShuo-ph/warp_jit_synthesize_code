"""Dot product kernel test using JAX."""
import jax
import jax.numpy as jnp

def dot_product(a, b):
    """Compute dot product using reduction."""
    return jnp.sum(a * b)

if __name__ == "__main__":
    n = 10
    a = jnp.array([float(i) for i in range(n)])  # 0, 1, 2, ..., 9
    b = jnp.array([float(i) for i in range(n)])  # 0, 1, 2, ..., 9
    
    # JIT compile the function
    dot_product_jit = jax.jit(dot_product)
    
    computed = dot_product_jit(a, b)
    expected = sum(i*i for i in range(n))  # 0 + 1 + 4 + 9 + ... = 285
    print(f"Dot product result: {computed}")
    print(f"Expected: {expected}")
    print(f"Match: {abs(computed - expected) < 1e-6}")
