import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.exp(x)
    v2 = jnp.sin(y)
    v3 = jnp.cos(v0)
    v4 = jnp.cos(v2)
    v5 = jnp.minimum(x, v1)
    return v5