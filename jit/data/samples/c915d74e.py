import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.cos(y)
    v2 = jnp.subtract(x, v1)
    v3 = jnp.add(v2, v1)
    v4 = jnp.exp(v0)
    v5 = jnp.sin(v1)
    return v5