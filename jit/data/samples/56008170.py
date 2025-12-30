import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.sin(y)
    v2 = jnp.subtract(x, v1)
    v3 = jnp.multiply(v0, v2)
    v4 = jnp.sin(v3)
    v5 = jnp.square(v2)
    v6 = jnp.minimum(v5, v2)
    v7 = jnp.tanh(v6)
    v8 = jnp.maximum(x, v3)
    v9 = jnp.subtract(v5, v2)
    return v9