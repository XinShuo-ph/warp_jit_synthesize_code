import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.subtract(v0, y)
    v2 = jnp.add(v0, y)
    v3 = jnp.multiply(v2, v0)
    v4 = jnp.exp(y)
    v5 = jnp.sin(v2)
    v6 = jnp.abs(x)
    v7 = jnp.minimum(v4, v4)
    v8 = jnp.add(v1, v0)
    return v8