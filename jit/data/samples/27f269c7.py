import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.maximum(x, y)
    v2 = jnp.add(v0, x)
    v3 = jnp.add(x, x)
    v4 = jnp.abs(v3)
    v5 = jnp.minimum(y, v3)
    v6 = jnp.tanh(v4)
    v7 = jnp.exp(v1)
    v8 = jnp.abs(v2)
    return v8