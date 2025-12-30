import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.abs(x)
    v2 = jnp.subtract(v1, y)
    v3 = jnp.maximum(y, x)
    v4 = jnp.square(v3)
    v5 = jnp.square(y)
    v6 = jnp.sin(v5)
    v7 = jnp.multiply(v1, v4)
    v8 = jnp.minimum(y, v6)
    v9 = jnp.maximum(v0, v7)
    return v9