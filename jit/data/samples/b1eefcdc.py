import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.minimum(x, v0)
    v2 = jnp.abs(y)
    v3 = jnp.abs(y)
    v4 = jnp.cos(v3)
    v5 = jnp.add(v3, v4)
    v6 = jnp.add(v5, v3)
    v7 = jnp.square(v2)
    v8 = jnp.square(v3)
    return v8