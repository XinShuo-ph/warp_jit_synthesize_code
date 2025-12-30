import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, x)
    v1 = jnp.multiply(x, x)
    v2 = jnp.exp(x)
    v3 = jnp.sin(v1)
    return v3