import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.subtract(v0, v0)
    v2 = jnp.minimum(v0, x)
    v3 = jnp.abs(v2)
    v4 = jnp.multiply(y, v0)
    v5 = jnp.multiply(v1, x)
    v6 = jnp.minimum(v3, v3)
    v7 = jnp.add(v1, v4)
    v8 = jnp.exp(y)
    v9 = jnp.sin(v2)
    return v9