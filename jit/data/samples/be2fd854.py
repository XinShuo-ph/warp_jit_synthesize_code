import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, y)
    v1 = jnp.exp(v0)
    v2 = jnp.add(v0, v1)
    v3 = jnp.add(x, x)
    v4 = jnp.maximum(v3, v0)
    return v4