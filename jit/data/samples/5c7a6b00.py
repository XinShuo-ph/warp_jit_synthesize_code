import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.square(x)
    v1 = jnp.abs(x)
    v2 = jnp.multiply(y, v1)
    v3 = jnp.multiply(y, v2)
    v4 = jnp.exp(v0)
    v5 = jnp.tanh(v0)
    return v5