import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.exp(v0)
    v2 = jnp.exp(x)
    v3 = jnp.exp(y)
    v4 = jnp.tanh(v3)
    v5 = jnp.cos(v2)
    v6 = jnp.exp(v1)
    v7 = jnp.multiply(v6, v3)
    v8 = jnp.multiply(v4, v0)
    v9 = jnp.square(v8)
    return v9