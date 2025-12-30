import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.subtract(x, x)
    v1 = jnp.tanh(x)
    v2 = jnp.maximum(x, x)
    v3 = jnp.add(x, v0)
    return v3