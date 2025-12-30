import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

x = jnp.array([1, 2, 3])
y = jnp.array([4, 5, 6])
z = x + y
print(f"Result of [1,2,3] + [4,5,6] = {z}")
