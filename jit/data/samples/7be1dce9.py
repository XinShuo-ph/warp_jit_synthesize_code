import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, y)
    v1 = jnp.square(v0)
    v2 = jnp.exp(x)
    return v2