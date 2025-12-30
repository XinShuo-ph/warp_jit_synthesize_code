import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.cos(x)
    v1 = jnp.tanh(v0)
    v2 = jnp.add(x, v0)
    return v2