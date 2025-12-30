import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(y)
    v1 = jnp.exp(v0)
    v2 = jnp.abs(v0)
    v3 = jnp.abs(v0)
    v4 = jnp.exp(v2)
    return v4