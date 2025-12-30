import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.subtract(y, y)
    v2 = jnp.subtract(x, v1)
    v3 = jnp.maximum(v0, x)
    v4 = jnp.maximum(v3, x)
    v5 = jnp.maximum(v1, x)
    v6 = jnp.sin(v3)
    v7 = jnp.multiply(v1, v6)
    v8 = jnp.minimum(v3, v1)
    return v8