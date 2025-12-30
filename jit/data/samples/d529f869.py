import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, y)
    v1 = jnp.maximum(x, v0)
    v2 = jnp.abs(x)
    v3 = jnp.square(v0)
    v4 = jnp.abs(v1)
    v5 = jnp.add(v0, v0)
    v6 = jnp.exp(v0)
    return v6