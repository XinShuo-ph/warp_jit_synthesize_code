import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(y, x)
    v1 = jnp.sin(y)
    v2 = jnp.add(y, v0)
    v3 = jnp.tanh(v2)
    v4 = jnp.maximum(y, v2)
    v5 = jnp.exp(v1)
    v6 = jnp.multiply(v0, v0)
    v7 = jnp.minimum(v6, v6)
    v8 = jnp.square(v1)
    v9 = jnp.subtract(v7, y)
    return v9