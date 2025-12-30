"""SAXPY (Single-precision A*X Plus Y) kernel test."""
import jax
import jax.numpy as jnp

def saxpy(a, x, y):
    """SAXPY: a * x + y"""
    return a * x + y

if __name__ == "__main__":
    n = 8
    a = 2.0
    x = jnp.array([float(i) for i in range(n)])
    y = jnp.array([float(i * 10) for i in range(n)])
    
    # JIT compile the function
    saxpy_jit = jax.jit(saxpy)
    
    result = saxpy_jit(a, x, y)
    expected = [a * i + i * 10 for i in range(n)]
    print(f"SAXPY result: {list(result)}")
    print(f"Expected: {expected}")
    print(f"Match: {all(abs(float(r) - e) < 1e-6 for r, e in zip(result, expected))}")
