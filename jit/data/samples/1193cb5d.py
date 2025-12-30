import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(y)
    v1 = jnp.tanh(v0)
    v2 = jnp.exp(v0)
    v3 = jnp.square(v1)
    v4 = jnp.subtract(v1, v0)
    v5 = jnp.tanh(y)
    v6 = jnp.subtract(v4, v3)
    v7 = jnp.minimum(x, v1)
    v8 = jnp.subtract(v2, v4)
    v9 = jnp.multiply(v2, y)
    return v9