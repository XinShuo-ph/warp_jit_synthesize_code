import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.square(v0)
    v2 = jnp.subtract(v0, y)
    v3 = jnp.maximum(v1, x)
    v4 = jnp.subtract(x, v2)
    v5 = jnp.exp(v0)
    return v5