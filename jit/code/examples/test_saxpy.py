"""SAXPY (Single-precision A*X Plus Y) kernel test."""
import jax
import jax.numpy as jnp

def saxpy(a, x, y):
    """SAXPY operation: out = a * x + y"""
    return a * x + y

if __name__ == "__main__":
    n = 8
    a = 2.0
    x = jnp.array([float(i) for i in range(n)])
    y = jnp.array([float(i * 10) for i in range(n)])
    
    # JIT compile
    jit_saxpy = jax.jit(saxpy)
    
    # Execute
    out = jit_saxpy(a, x, y)
    
    result = out
    expected = jnp.array([a * i + i * 10 for i in range(n)])
    print(f"SAXPY result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {jnp.allclose(result, expected)}")
