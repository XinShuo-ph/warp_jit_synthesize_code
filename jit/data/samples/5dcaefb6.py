import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.tanh(x)
    v1 = jnp.minimum(y, x)
    v2 = jnp.exp(x)
    v3 = jnp.cos(v2)
    return v3