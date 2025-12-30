import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, x)
    v1 = jnp.tanh(v0)
    v2 = jnp.cos(v1)
    v3 = jnp.subtract(v1, x)
    v4 = jnp.subtract(y, x)
    v5 = jnp.abs(v4)
    v6 = jnp.multiply(v0, v3)
    return v6