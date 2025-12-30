import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.exp(x)
    v2 = jnp.cos(y)
    v3 = jnp.exp(x)
    v4 = jnp.square(v2)
    v5 = jnp.abs(v3)
    return v5