import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.sin(x)
    v2 = jnp.exp(x)
    return v2