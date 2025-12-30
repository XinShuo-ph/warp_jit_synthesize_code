import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(y)
    v1 = jnp.tanh(x)
    v2 = jnp.minimum(v1, x)
    v3 = jnp.sin(v2)
    v4 = jnp.square(v0)
    v5 = jnp.maximum(v2, v0)
    v6 = jnp.maximum(v4, v2)
    v7 = jnp.minimum(v2, v6)
    v8 = jnp.maximum(v5, v7)
    return v8