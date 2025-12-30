import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, y)
    v1 = jnp.multiply(y, x)
    v2 = jnp.abs(v0)
    v3 = jnp.exp(v2)
    v4 = jnp.abs(v3)
    v5 = jnp.subtract(v2, v0)
    v6 = jnp.exp(v1)
    v7 = jnp.exp(v2)
    v8 = jnp.add(v6, y)
    return v8