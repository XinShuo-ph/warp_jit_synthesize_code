# JAX Function Compilation Flow

## Python → HLO/XLA Pipeline

1. **Function Definition**: Regular Python function with JAX numpy
2. **Tracing**: `jax.jit` traces the function with abstract values
3. **Jaxpr Generation**: Traces produce Jaxpr (JAX Program Representation)
4. **HLO Lowering**: Jaxpr is lowered to HLO (XLA's IR)
5. **XLA Compilation**: HLO is optimized and compiled to machine code
6. **Caching**: Results cached for reuse with same input shapes/types

## Key Components

- `jax.jit`: JIT compilation decorator
- `jax.make_jaxpr`: Extract Jaxpr representation
- `jax.grad`: Automatic differentiation
- `jax.vmap`: Vectorization transformation

## IR Access Methods

```python
import jax
import jax.numpy as jnp

def func(a, b):
    return a + b

# Sample inputs for tracing
a = jnp.ones((100,))
b = jnp.ones((100,))

# 1. Get Jaxpr (high-level IR)
jaxpr = jax.make_jaxpr(func)(a, b)
print("Jaxpr:", jaxpr)

# 2. Get lowered HLO (before optimization)
lowered = jax.jit(func).lower(a, b)
hlo_text = lowered.as_text()
print("HLO:", hlo_text)

# 3. Get compiled HLO (after optimization)
compiled = lowered.compile()
optimized_hlo = compiled.as_text()
print("Optimized HLO:", optimized_hlo)
```

## Forward/Backward Code

JAX generates backward (gradient) code via automatic differentiation:

```python
# Forward function
def func(a):
    return jnp.sum(a ** 2)

# Backward (gradient) function
grad_func = jax.grad(func)

# Get gradient Jaxpr
grad_jaxpr = jax.make_jaxpr(grad_func)(a)
```

## Transformations

JAX supports composable function transformations:

| Transform | Purpose |
|-----------|---------|
| `jax.jit` | JIT compile to XLA |
| `jax.grad` | Automatic differentiation |
| `jax.vmap` | Automatic vectorization |
| `jax.pmap` | Parallel mapping across devices |
| `jax.value_and_grad` | Get value and gradient together |

## Device Placement

```python
# CPU execution
with jax.default_device(jax.devices('cpu')[0]):
    result = jax.jit(func)(a, b)

# GPU execution (if available)
with jax.default_device(jax.devices('gpu')[0]):
    result = jax.jit(func)(a, b)
```

## Cache Location

JAX caches compiled code in memory by default. Persistent caching can be enabled:

```python
jax.config.update("jax_compilation_cache_dir", "/path/to/cache")
```

## Tracing Rules

JAX tracing is based on input shapes and dtypes:
- Same shapes/dtypes → reuse cached compilation
- Different shapes → recompile
- Dynamic shapes → use `jax.pure_callback` or shape polymorphism
