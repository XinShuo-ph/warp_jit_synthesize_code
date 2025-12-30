import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.tanh(v0)
    v2 = jnp.sin(y)
    v3 = jnp.sin(x)
    v4 = jnp.minimum(v1, v3)
    v5 = jnp.minimum(v2, v0)
    v6 = jnp.sin(v0)
    v7 = jnp.multiply(v1, y)
    v8 = jnp.abs(v5)
    return v8