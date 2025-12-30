import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, y)
    v1 = jnp.maximum(y, x)
    v2 = jnp.sin(v1)
    v3 = jnp.exp(y)
    v4 = jnp.square(x)
    v5 = jnp.subtract(v2, v3)
    v6 = jnp.cos(v2)
    v7 = jnp.minimum(v2, v0)
    v8 = jnp.minimum(v0, v6)
    return v8