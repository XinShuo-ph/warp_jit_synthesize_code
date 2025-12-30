import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.sin(x)
    v2 = jnp.exp(v1)
    v3 = jnp.cos(y)
    v4 = jnp.sin(v2)
    v5 = jnp.multiply(v4, v1)
    v6 = jnp.sin(v2)
    return v6