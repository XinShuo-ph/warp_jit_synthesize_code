import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(x, x)
    v1 = jnp.minimum(x, x)
    v2 = jnp.add(y, v1)
    v3 = jnp.multiply(v0, v2)
    return v3