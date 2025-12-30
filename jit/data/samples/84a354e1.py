import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.exp(v0)
    v2 = jnp.minimum(v1, v0)
    return v2