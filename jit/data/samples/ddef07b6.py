import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, x)
    v1 = jnp.tanh(x)
    v2 = jnp.square(y)
    v3 = jnp.subtract(v2, y)
    v4 = jnp.minimum(v0, y)
    v5 = jnp.square(v1)
    v6 = jnp.minimum(v1, v0)
    v7 = jnp.tanh(x)
    v8 = jnp.cos(y)
    v9 = jnp.sin(v8)
    return v9