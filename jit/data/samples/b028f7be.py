import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(y)
    v1 = jnp.minimum(v0, x)
    v2 = jnp.maximum(x, y)
    v3 = jnp.add(v0, v2)
    v4 = jnp.exp(v0)
    v5 = jnp.minimum(x, y)
    v6 = jnp.exp(v5)
    return v6