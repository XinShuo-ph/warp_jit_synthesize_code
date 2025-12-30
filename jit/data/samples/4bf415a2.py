import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(y, x)
    v1 = jnp.square(x)
    v2 = jnp.square(x)
    v3 = jnp.tanh(v1)
    v4 = jnp.multiply(v3, y)
    v5 = jnp.maximum(v3, v3)
    v6 = jnp.sin(x)
    return v6