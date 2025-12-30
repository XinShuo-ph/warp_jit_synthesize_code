import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.add(y, v0)
    v2 = jnp.sin(x)
    v3 = jnp.maximum(v2, v0)
    v4 = jnp.tanh(v2)
    v5 = jnp.subtract(y, v3)
    v6 = jnp.multiply(v4, v0)
    v7 = jnp.multiply(v1, x)
    v8 = jnp.maximum(v2, v4)
    return v8