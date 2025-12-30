import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.exp(v0)
    v2 = jnp.minimum(x, v1)
    v3 = jnp.minimum(x, y)
    return v3