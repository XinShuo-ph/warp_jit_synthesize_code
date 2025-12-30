import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.multiply(y, y)
    v1 = jnp.exp(x)
    v2 = jnp.subtract(y, v0)
    v3 = jnp.exp(v0)
    v4 = jnp.tanh(x)
    v5 = jnp.subtract(v4, x)
    v6 = jnp.cos(v4)
    v7 = jnp.square(v6)
    v8 = jnp.cos(v0)
    return v8