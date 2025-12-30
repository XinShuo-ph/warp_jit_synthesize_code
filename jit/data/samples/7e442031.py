import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.square(y)
    v2 = jnp.square(y)
    v3 = jnp.multiply(v1, v2)
    v4 = jnp.maximum(y, y)
    v5 = jnp.exp(v2)
    v6 = jnp.exp(v3)
    return v6