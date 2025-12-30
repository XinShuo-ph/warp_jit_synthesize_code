import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.minimum(v0, y)
    v2 = jnp.tanh(x)
    return v2