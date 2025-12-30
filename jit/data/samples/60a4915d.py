import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.exp(v0)
    v2 = jnp.add(x, x)
    v3 = jnp.cos(x)
    v4 = jnp.maximum(v2, v1)
    v5 = jnp.maximum(v4, x)
    v6 = jnp.multiply(v5, v1)
    v7 = jnp.add(v4, v6)
    v8 = jnp.minimum(v7, v5)
    v9 = jnp.multiply(v6, x)
    return v9