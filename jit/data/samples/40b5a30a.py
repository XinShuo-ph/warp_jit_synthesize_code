import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.exp(y)
    v2 = jnp.maximum(v1, v0)
    v3 = jnp.minimum(x, x)
    v4 = jnp.tanh(v1)
    v5 = jnp.abs(v3)
    v6 = jnp.square(v1)
    return v6