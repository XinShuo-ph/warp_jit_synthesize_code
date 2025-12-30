import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.subtract(y, x)
    v2 = jnp.square(y)
    v3 = jnp.abs(v0)
    v4 = jnp.square(v3)
    v5 = jnp.minimum(v1, v3)
    return v5