import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(y)
    v1 = jnp.subtract(v0, y)
    v2 = jnp.tanh(v0)
    v3 = jnp.exp(x)
    v4 = jnp.cos(v3)
    v5 = jnp.subtract(x, v0)
    v6 = jnp.multiply(v4, v2)
    v7 = jnp.maximum(v2, v6)
    v8 = jnp.tanh(v6)
    return v8