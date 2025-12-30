import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.exp(y)
    v2 = jnp.multiply(x, x)
    v3 = jnp.sin(x)
    v4 = jnp.subtract(y, y)
    v5 = jnp.multiply(v1, x)
    v6 = jnp.add(v2, v3)
    v7 = jnp.tanh(v3)
    v8 = jnp.sin(v0)
    return v8