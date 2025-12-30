import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.exp(y)
    v2 = jnp.abs(x)
    v3 = jnp.abs(y)
    v4 = jnp.sin(v1)
    v5 = jnp.multiply(v0, v0)
    v6 = jnp.exp(v2)
    return v6