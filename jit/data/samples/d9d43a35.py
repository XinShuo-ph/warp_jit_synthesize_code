import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.multiply(x, x)
    v2 = jnp.minimum(v0, v0)
    v3 = jnp.cos(y)
    v4 = jnp.minimum(v0, v3)
    v5 = jnp.cos(x)
    return v5