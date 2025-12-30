import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(y)
    v1 = jnp.cos(v0)
    v2 = jnp.cos(v0)
    v3 = jnp.subtract(v2, v1)
    v4 = jnp.maximum(v3, v3)
    v5 = jnp.square(x)
    v6 = jnp.sin(y)
    v7 = jnp.cos(v1)
    v8 = jnp.add(v3, v1)
    return v8