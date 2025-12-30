import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(x)
    v1 = jnp.sin(v0)
    v2 = jnp.cos(y)
    v3 = jnp.add(v2, v2)
    v4 = jnp.multiply(v0, v0)
    return v4