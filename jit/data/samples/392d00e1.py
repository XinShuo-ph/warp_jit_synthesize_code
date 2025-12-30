import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, y)
    v1 = jnp.square(v0)
    v2 = jnp.exp(y)
    v3 = jnp.square(x)
    v4 = jnp.minimum(v0, y)
    v5 = jnp.maximum(v0, y)
    v6 = jnp.minimum(y, v0)
    v7 = jnp.maximum(x, v0)
    v8 = jnp.add(v3, v5)
    v9 = jnp.multiply(v4, x)
    return v9