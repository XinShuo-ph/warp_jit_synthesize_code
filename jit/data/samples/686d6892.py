import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.exp(x)
    v2 = jnp.add(v0, y)
    v3 = jnp.square(v0)
    v4 = jnp.sin(x)
    v5 = jnp.sin(v4)
    v6 = jnp.exp(v0)
    v7 = jnp.cos(v4)
    v8 = jnp.sin(y)
    v9 = jnp.sin(v4)
    return v9