import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.exp(y)
    v1 = jnp.square(v0)
    v2 = jnp.add(v1, y)
    return v2