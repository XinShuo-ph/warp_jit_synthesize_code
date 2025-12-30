import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.exp(y)
    v2 = jnp.subtract(x, x)
    v3 = jnp.minimum(v0, v1)
    v4 = jnp.sin(v0)
    return v4