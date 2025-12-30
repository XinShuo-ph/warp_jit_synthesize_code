import jax
import jax.numpy as jnp

def generated_fn(x, y):
    v0 = jnp.add(x, x)
    v1 = jnp.multiply(v0, v0)
    v2 = jnp.tanh(v0)
    v3 = jnp.multiply(y, v1)
    v4 = jnp.sin(v2)
    v5 = jnp.abs(v3)
    v6 = jnp.add(v1, v4)
    v7 = jnp.tanh(v0)
    v8 = jnp.subtract(x, v0)
    return v8