import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.cos(x)
    v2 = jnp.subtract(v1, v1)
    v3 = jnp.exp(v1)
    v4 = jnp.minimum(v1, v0)
    v5 = jnp.exp(y)
    v6 = jnp.add(v2, y)
    return v6