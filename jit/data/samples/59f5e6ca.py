import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.sin(v0)
    v2 = jnp.multiply(x, x)
    v3 = jnp.maximum(x, v1)
    v4 = jnp.tanh(v3)
    return v4