import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.exp(v0)
    v2 = jnp.add(v0, y)
    v3 = jnp.multiply(x, x)
    v4 = jnp.multiply(v0, y)
    v5 = jnp.add(x, v1)
    v6 = jnp.maximum(v0, y)
    v7 = jnp.square(v0)
    v8 = jnp.sin(v3)
    v9 = jnp.cos(v2)
    return v9