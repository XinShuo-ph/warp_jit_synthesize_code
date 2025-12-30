import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.exp(x)
    v2 = jnp.tanh(v1)
    v3 = jnp.multiply(x, v2)
    v4 = jnp.cos(y)
    v5 = jnp.minimum(v4, v0)
    v6 = jnp.exp(v0)
    return v6