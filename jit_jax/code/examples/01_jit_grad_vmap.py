import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
import time

def selu(x, alpha=1.67, lmbda=1.05):
    return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

# 1. JIT Compilation
selu_jit = jit(selu)

x = jnp.arange(1000000, dtype=jnp.float32)

# Warmup
selu_jit(x).block_until_ready()

start = time.time()
selu_jit(x).block_until_ready()
end = time.time()
print(f"JIT execution time: {end - start:.6f} s")

# Non-JIT
start = time.time()
selu(x).block_until_ready()
end = time.time()
print(f"Python execution time: {end - start:.6f} s")


# 2. Gradient
def sum_logistic(x):
    return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x_small = jnp.arange(3.0)
derivative_fn = grad(sum_logistic)
print(f"Gradient at {x_small}: {derivative_fn(x_small)}")

# 3. VMAP
mat = jnp.arange(9).reshape(3, 3)
vec = jnp.arange(3)

def dot(v1, v2):
    return jnp.dot(v1, v2)

# Vectorized dot product (matrix-vector multiplication)
vmap_dot = vmap(dot, in_axes=(0, None))
result = vmap_dot(mat, vec)
print(f"VMAP (MatMul) result: {result}")
