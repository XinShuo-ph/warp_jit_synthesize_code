import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.exp(v0)
    v2 = jnp.square(v0)
    v3 = jnp.cos(x)
    v4 = jnp.minimum(y, v1)
    v5 = jnp.abs(v4)
    v6 = jnp.cos(v0)
    v7 = jnp.minimum(v6, v3)
    v8 = jnp.add(v4, v5)
    return v8