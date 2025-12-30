"""Simple JAX kernel test."""
import jax
import jax.numpy as jnp

def add_kernel(a, b):
    """Element-wise addition kernel."""
    return a + b

if __name__ == "__main__":
    n = 10
    a = jnp.array([float(i) for i in range(n)])
    b = jnp.array([float(i) for i in range(n)])
    
    # JIT compile the function
    add_kernel_jit = jax.jit(add_kernel)
    
    c = add_kernel_jit(a, b)
    print("Result:", c)
    print("Expected:", [float(i*2) for i in range(n)])
    print("Kernel compiled and executed successfully!")
