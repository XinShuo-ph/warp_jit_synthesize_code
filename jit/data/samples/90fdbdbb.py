import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, y)
    v1 = jnp.subtract(x, y)
    v2 = jnp.minimum(x, v0)
    v3 = jnp.add(v0, v0)
    v4 = jnp.cos(y)
    v5 = jnp.multiply(y, y)
    v6 = jnp.multiply(v2, v1)
    v7 = jnp.add(v4, v6)
    v8 = jnp.square(v4)
    v9 = jnp.square(v1)
    return v9