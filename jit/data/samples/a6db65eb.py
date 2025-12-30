import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(y)
    v1 = jnp.multiply(v0, y)
    v2 = jnp.exp(x)
    v3 = jnp.exp(v2)
    v4 = jnp.subtract(v2, v1)
    return v4