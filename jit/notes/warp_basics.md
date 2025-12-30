# JAX Basics

## Overview

JAX is a library for high-performance numerical computing with automatic differentiation and JIT compilation via XLA.

## Key Concepts

### 1. JIT Compilation

JAX can compile functions to optimized machine code:

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_function(x, y):
    return x * y + jnp.sin(x)

# First call compiles
result = my_function(jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
# Subsequent calls use cached compiled version
```

### 2. Automatic Differentiation

JAX provides automatic gradient computation:

```python
# Define a function
def loss_fn(x):
    return jnp.sum(x ** 2)

# Get gradient function
grad_fn = jax.grad(loss_fn)

# Compute gradient
x = jnp.array([1.0, 2.0, 3.0])
gradient = grad_fn(x)  # [2.0, 4.0, 6.0]
```

### 3. Vectorization

JAX can automatically vectorize functions:

```python
# Apply function to batch of inputs
batched_fn = jax.vmap(my_function)
results = batched_fn(batch_of_x, batch_of_y)
```

## Compilation Flow

1. **Python Function** - Written using JAX/NumPy API
2. **Tracing** - JAX traces the function with abstract values
3. **Jaxpr** - Creates JAX's internal intermediate representation
4. **HLO** - Converts to XLA's HLO (High-Level Optimizer) IR
5. **XLA Optimization** - Applies optimization passes
6. **Target Code** - Generates CPU/GPU/TPU code

## XLA Backend

JAX uses XLA (Accelerated Linear Algebra) for compilation:

- **CPU**: Generates optimized CPU code (LLVM)
- **GPU**: Generates CUDA kernels (NVIDIA) or ROCm (AMD)
- **TPU**: Generates TPU-specific code (Google Cloud)

## Core Transformations

### grad - Differentiation
```python
grad_fn = jax.grad(func)  # Derivative with respect to first argument
grad_fn = jax.grad(func, argnums=(0, 1))  # Multiple arguments
```

### jit - Just-In-Time Compilation
```python
fast_fn = jax.jit(func)
```

### vmap - Vectorization
```python
batched_fn = jax.vmap(func)
```

### pmap - Parallelization
```python
parallel_fn = jax.pmap(func)  # Parallel across devices
```

## JAX Array Operations

JAX provides a NumPy-compatible API via `jax.numpy`:

```python
import jax.numpy as jnp

# Create arrays
a = jnp.array([1.0, 2.0, 3.0])
b = jnp.zeros((3, 3))

# Operations
c = jnp.dot(a, a)
d = jnp.sum(a)
e = jnp.sin(a)
```

## Control Flow

JAX provides functional control flow primitives:

```python
# Conditional
result = jax.lax.cond(predicate, true_fn, false_fn, operand)

# While loop
result = jax.lax.while_loop(cond_fn, body_fn, init_val)

# For loop
result = jax.lax.fori_loop(0, n, body_fn, init_val)

# Scan (for sequences)
final, outputs = jax.lax.scan(body_fn, init, xs)
```

## Key Differences from NumPy

1. **Immutability**: JAX arrays are immutable
   ```python
   # This won't work:
   a[0] = 5  # Error!
   
   # Instead:
   a = a.at[0].set(5)
   ```

2. **Pure Functions**: Functions should be side-effect free for JIT

3. **Static Arguments**: Some values must be known at compile time
   ```python
   @jax.jit
   def loop(x, n):
       return jax.lax.fori_loop(0, n, lambda i, x: x + 1, x)
   
   # n must be a concrete integer, not a traced value
   ```

## Random Numbers

JAX uses explicit random state:

```python
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)
random_values = jax.random.normal(subkey, shape=(10,))
```

## Debugging

```python
# Print during tracing
jax.debug.print("Value: {}", x)

# Prevent compilation for debugging
with jax.disable_jit():
    result = my_function(x)
```

## Performance Tips

1. Use `@jax.jit` on performance-critical functions
2. Avoid Python loops - use `jax.lax.scan` or `jax.lax.fori_loop`
3. Use `vmap` for batching instead of manual loops
4. Keep functions pure (no side effects)
5. Use static arguments when possible

## Resources

- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX GitHub](https://github.com/google/jax)
- [XLA Documentation](https://www.tensorflow.org/xla)
