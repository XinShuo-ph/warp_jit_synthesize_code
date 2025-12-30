import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.abs(v0)
    v2 = jnp.add(y, y)
    v3 = jnp.subtract(v0, x)
    v4 = jnp.multiply(x, y)
    v5 = jnp.sin(v0)
    v6 = jnp.add(v4, v5)
    v7 = jnp.square(y)
    v8 = jnp.cos(v6)
    return v8