import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.minimum(x, x)
    v2 = jnp.exp(v1)
    v3 = jnp.subtract(v2, v0)
    return v3