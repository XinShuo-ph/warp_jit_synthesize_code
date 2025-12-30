import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.cos(v0)
    v2 = jnp.abs(x)
    return v2