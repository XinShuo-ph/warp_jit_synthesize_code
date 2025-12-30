import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, y)
    v1 = jnp.square(v0)
    v2 = jnp.sin(y)
    v3 = jnp.add(v0, y)
    v4 = jnp.cos(x)
    v5 = jnp.cos(v0)
    v6 = jnp.exp(v3)
    return v6