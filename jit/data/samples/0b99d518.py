import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, x)
    v1 = jnp.multiply(v0, v0)
    v2 = jnp.add(y, v0)
    v3 = jnp.subtract(v1, y)
    v4 = jnp.add(v1, x)
    v5 = jnp.cos(v3)
    v6 = jnp.abs(v1)
    v7 = jnp.tanh(v1)
    v8 = jnp.minimum(v4, v6)
    v9 = jnp.add(v1, y)
    return v9