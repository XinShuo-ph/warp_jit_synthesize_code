import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.maximum(y, y)
    v2 = jnp.exp(v1)
    v3 = jnp.add(v0, v0)
    v4 = jnp.tanh(v2)
    v5 = jnp.maximum(v4, v2)
    v6 = jnp.abs(v4)
    v7 = jnp.square(x)
    v8 = jnp.multiply(v3, x)
    return v8