import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, x)
    v1 = jnp.multiply(x, x)
    v2 = jnp.exp(x)
    v3 = jnp.sin(v1)
    v4 = jnp.subtract(v3, v2)
    v5 = jnp.subtract(x, v2)
    v6 = jnp.sin(v0)
    return v6