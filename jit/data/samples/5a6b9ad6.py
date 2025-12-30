import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.exp(v0)
    v2 = jnp.add(v0, v0)
    v3 = jnp.subtract(v2, v1)
    v4 = jnp.multiply(y, x)
    return v4