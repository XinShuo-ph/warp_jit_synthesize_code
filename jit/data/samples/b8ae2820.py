import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.sin(x)
    v1 = jnp.sin(v0)
    v2 = jnp.abs(v0)
    v3 = jnp.subtract(y, v2)
    v4 = jnp.tanh(v0)
    v5 = jnp.multiply(v3, y)
    return v5