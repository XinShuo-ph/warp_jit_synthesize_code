import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.sin(v0)
    v2 = jnp.maximum(x, x)
    v3 = jnp.subtract(v2, x)
    v4 = jnp.abs(x)
    v5 = jnp.tanh(v1)
    v6 = jnp.add(v4, v2)
    return v6