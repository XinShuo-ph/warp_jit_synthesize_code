import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.subtract(x, v0)
    v2 = jnp.square(v1)
    v3 = jnp.exp(v2)
    v4 = jnp.multiply(v0, v1)
    return v4