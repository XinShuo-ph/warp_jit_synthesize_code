import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, y)
    v1 = jnp.sin(y)
    v2 = jnp.exp(x)
    v3 = jnp.minimum(y, v0)
    v4 = jnp.exp(v3)
    return v4