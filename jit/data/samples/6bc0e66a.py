import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.multiply(y, v0)
    v2 = jnp.multiply(v0, v1)
    v3 = jnp.cos(x)
    v4 = jnp.square(v2)
    return v4