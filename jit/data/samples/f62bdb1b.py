import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(y)
    v1 = jnp.square(x)
    v2 = jnp.subtract(v1, y)
    v3 = jnp.add(v0, v1)
    v4 = jnp.maximum(v0, y)
    v5 = jnp.add(x, v4)
    v6 = jnp.tanh(v0)
    v7 = jnp.sin(v1)
    v8 = jnp.tanh(v0)
    v9 = jnp.minimum(v1, v5)
    return v9