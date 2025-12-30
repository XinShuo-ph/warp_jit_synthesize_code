import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(x, x)
    v1 = jnp.square(y)
    v2 = jnp.add(x, y)
    v3 = jnp.exp(x)
    return v3