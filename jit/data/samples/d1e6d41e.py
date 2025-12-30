import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.subtract(y, v0)
    v2 = jnp.cos(v1)
    v3 = jnp.multiply(v1, x)
    return v3