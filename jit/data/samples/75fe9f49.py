import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, y)
    v1 = jnp.sin(x)
    v2 = jnp.minimum(v1, v1)
    v3 = jnp.square(x)
    v4 = jnp.exp(v2)
    v5 = jnp.exp(x)
    return v5