import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.maximum(x, x)
    v1 = jnp.multiply(v0, v0)
    v2 = jnp.exp(x)
    return v2