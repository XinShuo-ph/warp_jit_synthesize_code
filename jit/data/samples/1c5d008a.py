import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, y)
    v1 = jnp.square(y)
    v2 = jnp.exp(x)
    v3 = jnp.minimum(v1, v1)
    v4 = jnp.exp(v3)
    v5 = jnp.sin(v0)
    return v5