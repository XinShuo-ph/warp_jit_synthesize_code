import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, y)
    v1 = jnp.tanh(y)
    v2 = jnp.square(v1)
    return v2