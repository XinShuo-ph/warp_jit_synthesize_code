import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.cos(v0)
    v2 = jnp.square(v1)
    v3 = jnp.abs(y)
    return v3