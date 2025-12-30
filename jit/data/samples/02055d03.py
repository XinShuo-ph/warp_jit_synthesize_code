import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.abs(y)
    v2 = jnp.maximum(x, v0)
    v3 = jnp.exp(v1)
    v4 = jnp.tanh(x)
    v5 = jnp.cos(v4)
    return v5