import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(y, x)
    v1 = jnp.minimum(v0, v0)
    v2 = jnp.cos(v1)
    v3 = jnp.maximum(x, x)
    v4 = jnp.add(v2, v3)
    v5 = jnp.tanh(x)
    v6 = jnp.square(y)
    v7 = jnp.exp(v1)
    v8 = jnp.subtract(v1, v1)
    v9 = jnp.add(x, y)
    return v9