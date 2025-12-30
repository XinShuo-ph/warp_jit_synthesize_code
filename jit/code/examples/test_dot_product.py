"""Dot product kernel test using reduction."""
import jax
import jax.numpy as jnp

def dot_product(a, b):
    """Dot product: sum of element-wise products"""
    return jnp.sum(a * b)

if __name__ == "__main__":
    n = 10
    a = jnp.array([float(i) for i in range(n)])  # 0, 1, 2, ..., 9
    b = jnp.array([float(i) for i in range(n)])  # 0, 1, 2, ..., 9
    
    # JIT compile
    jit_dot = jax.jit(dot_product)
    
    # Execute
    result = jit_dot(a, b)
    
    computed = float(result)
    expected = sum(i*i for i in range(n))  # 0 + 1 + 4 + 9 + ... = 285
    print(f"Dot product result: {computed}")
    print(f"Expected: {expected}")
    print(f"Match: {abs(computed - expected) < 1e-6}")
