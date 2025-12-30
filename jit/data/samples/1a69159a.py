import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, x)
    v1 = jnp.square(y)
    v2 = jnp.minimum(v0, x)
    v3 = jnp.subtract(v2, v2)
    v4 = jnp.exp(y)
    v5 = jnp.minimum(y, y)
    v6 = jnp.minimum(v0, y)
    v7 = jnp.tanh(v2)
    v8 = jnp.subtract(v6, v4)
    return v8