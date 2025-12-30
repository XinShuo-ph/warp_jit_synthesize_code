import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, y)
    v1 = jnp.add(y, x)
    v2 = jnp.minimum(v1, x)
    v3 = jnp.exp(x)
    v4 = jnp.exp(y)
    v5 = jnp.sin(v4)
    return v5