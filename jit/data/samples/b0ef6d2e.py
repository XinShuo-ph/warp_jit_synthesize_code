import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.minimum(y, y)
    v1 = jnp.tanh(y)
    v2 = jnp.add(v1, v0)
    return v2