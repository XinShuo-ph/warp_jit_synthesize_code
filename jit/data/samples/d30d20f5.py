import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, y)
    v1 = jnp.square(x)
    v2 = jnp.minimum(x, x)
    v3 = jnp.multiply(v0, y)
    v4 = jnp.tanh(v1)
    v5 = jnp.abs(v2)
    v6 = jnp.subtract(v5, v3)
    v7 = jnp.sin(v3)
    v8 = jnp.multiply(v5, y)
    v9 = jnp.minimum(v5, v2)
    return v9