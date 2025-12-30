import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, y)
    v1 = jnp.sin(v0)
    v2 = jnp.subtract(v1, v0)
    v3 = jnp.abs(y)
    v4 = jnp.cos(v0)
    v5 = jnp.tanh(x)
    v6 = jnp.subtract(v3, v5)
    v7 = jnp.multiply(v0, y)
    v8 = jnp.abs(v5)
    v9 = jnp.add(v1, v3)
    return v9