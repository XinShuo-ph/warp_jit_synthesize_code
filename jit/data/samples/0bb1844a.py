import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.abs(x)
    v1 = jnp.abs(y)
    v2 = jnp.add(y, v0)
    v3 = jnp.cos(v1)
    v4 = jnp.subtract(v0, y)
    v5 = jnp.multiply(v0, v0)
    v6 = jnp.tanh(v3)
    return v6