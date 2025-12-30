"""Simple JAX kernel test."""
import jax
import jax.numpy as jnp

def add_kernel(a, b):
    """Elementwise addition: c = a + b"""
    return a + b

if __name__ == "__main__":
    n = 10
    a = jnp.array([float(i) for i in range(n)])
    b = jnp.array([float(i) for i in range(n)])
    
    # JIT compile the function
    jit_add = jax.jit(add_kernel)
    
    # Execute
    c = jit_add(a, b)
    
    print("Result:", c)
    print("Expected:", [float(i*2) for i in range(n)])
    print("Match:", jnp.allclose(c, jnp.array([float(i*2) for i in range(n)])))
    print("Kernel compiled and executed successfully!")
