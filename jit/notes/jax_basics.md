# JAX Basics

## Overview

JAX is a high-performance numerical computing library that combines NumPy-like APIs with:
- **JIT compilation** via XLA (Accelerated Linear Algebra)
- **Automatic differentiation** (forward and reverse mode)
- **Vectorization** (vmap) and parallelization (pmap)

## Core Transformations

### JIT Compilation

```python
import jax
import jax.numpy as jnp

@jax.jit
def add_arrays(a, b):
    return a + b

# First call traces and compiles
result = add_arrays(jnp.ones(100), jnp.ones(100))

# Subsequent calls use cached compiled code
result = add_arrays(jnp.ones(100), jnp.ones(100))
```

### Automatic Differentiation

```python
# Gradient of scalar-valued function
def loss(x):
    return jnp.sum(x ** 2)

grad_loss = jax.grad(loss)
gradient = grad_loss(jnp.array([1.0, 2.0, 3.0]))

# Value and gradient together
val, grad = jax.value_and_grad(loss)(jnp.array([1.0, 2.0, 3.0]))
```

### Vectorization (vmap)

```python
# Single-element function
def process_element(x):
    return jnp.sin(x) + jnp.cos(x)

# Automatically vectorize over batch dimension
batch_process = jax.vmap(process_element)
result = batch_process(jnp.ones(100))
```

## Compilation Pipeline

```
Python Function
      │
      ▼ (trace)
   Jaxpr (JAX Expression)
      │
      ▼ (lower)
   HLO (High-Level Optimizer)
      │
      ▼ (compile)
   Optimized XLA Executable
      │
      ▼ (execute)
   Hardware (CPU/GPU/TPU)
```

## Key Differences from NumPy

| Aspect | NumPy | JAX |
|--------|-------|-----|
| Mutability | Mutable arrays | Immutable arrays |
| Random | Stateful | Explicit key passing |
| JIT | No | Yes (XLA) |
| Gradients | No | Yes (autodiff) |
| GPU/TPU | Limited | Native support |

## Common Patterns

### Elementwise Operations
```python
def elementwise(a, b):
    return a + b * 2.0
```

### Reductions
```python
def reduce_sum(a):
    return jnp.sum(a)

def reduce_mean_axis(a):
    return jnp.mean(a, axis=-1)
```

### Conditionals (jnp.where)
```python
def conditional(a, threshold):
    return jnp.where(a > threshold, a * 2, a * 0.5)
```

### Matrix Operations
```python
def matmul(a, b):
    return jnp.matmul(a, b)

def batch_matmul(a, b):
    return jnp.einsum('bij,bjk->bik', a, b)
```

## Extracting IR

### Get Jaxpr
```python
from jax import make_jaxpr

def func(a, b):
    return a + b

jaxpr = make_jaxpr(func)(jnp.ones(10), jnp.ones(10))
print(jaxpr)
```

### Get HLO
```python
lowered = jax.jit(func).lower(jnp.ones(10), jnp.ones(10))
print(lowered.as_text())
```

### Get Compiled Info
```python
compiled = lowered.compile()
print(compiled.cost_analysis())
```

## Best Practices

1. **Use jax.numpy instead of numpy** for JAX compatibility
2. **Avoid Python control flow** in JIT-compiled functions (use `jax.lax.cond`, `jax.lax.scan`)
3. **Use pure functions** without side effects
4. **Handle random state explicitly** with PRNGKey
5. **Prefer immutable operations** (no in-place updates)

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub](https://github.com/google/jax)
- [XLA Overview](https://www.tensorflow.org/xla)
