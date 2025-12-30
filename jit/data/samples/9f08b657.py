import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, x)
    v1 = jnp.multiply(v0, x)
    v2 = jnp.minimum(v1, x)
    v3 = jnp.add(v1, v0)
    v4 = jnp.sin(y)
    return v4