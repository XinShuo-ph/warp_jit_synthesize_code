import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.exp(y)
    v2 = jnp.maximum(y, x)
    v3 = jnp.sin(v2)
    v4 = jnp.tanh(v1)
    return v4