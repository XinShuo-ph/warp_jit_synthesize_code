import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, x)
    v1 = jnp.add(y, v0)
    v2 = jnp.minimum(v1, v1)
    v3 = jnp.minimum(x, y)
    v4 = jnp.tanh(v3)
    v5 = jnp.subtract(v0, v0)
    v6 = jnp.add(v5, v4)
    v7 = jnp.add(v1, y)
    return v7