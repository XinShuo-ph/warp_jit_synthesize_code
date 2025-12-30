import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.multiply(v0, x)
    v2 = jnp.abs(v0)
    v3 = jnp.abs(x)
    v4 = jnp.multiply(v0, y)
    v5 = jnp.add(x, y)
    v6 = jnp.sin(v1)
    v7 = jnp.square(v0)
    v8 = jnp.sin(v6)
    return v8