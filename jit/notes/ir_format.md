# JAX IR (HLO/Jaxpr) Format

## Structure

JAX provides multiple levels of intermediate representation:

1. **Jaxpr**: High-level functional representation
2. **HLO**: XLA's High Level Optimizer representation
3. **Optimized HLO**: After XLA compilation optimizations

## Jaxpr Format

Jaxpr (JAX Program Representation) is a functional IR:

```
{ lambda ; a:f32[100] b:f32[100]. let
    c:f32[100] = add a b
    d:f32[100] = mul c 2.0
  in (d,) }
```

Components:
- `lambda`: Input bindings with types
- `let`: Body with named intermediate values
- `in`: Output tuple

## HLO Format

HLO (High Level Optimizer) is XLA's IR:

```
HloModule jit_func, entry_computation_layout={(f32[100]{0})->f32[100]{0}}

ENTRY main.3 {
  Arg_0.1 = f32[100]{0} parameter(0)
  constant.2 = f32[] constant(2)
  ROOT multiply.3 = f32[100]{0} multiply(Arg_0.1, constant.2)
}
```

Components:
- `HloModule`: Module declaration with entry layout
- `ENTRY`: Main computation entry point
- Instructions with SSA naming (`.N` suffix)
- Type annotations with layout `{0}` or `{1,0}`

## Key Patterns

| Python | Jaxpr | HLO |
|--------|-------|-----|
| `a + b` | `add a b` | `add(Arg_0.1, Arg_1.2)` |
| `a * b` | `mul a b` | `multiply(Arg_0.1, Arg_1.2)` |
| `jnp.sin(a)` | `sin a` | `sine(Arg_0.1)` |
| `jnp.where(c, a, b)` | `select c a b` | `select(Cond.1, True.2, False.3)` |
| `jnp.sum(a)` | `reduce_sum a` | `reduce(Arg.1, ...)` |

## Backward (Gradient) Code

JAX automatically generates gradient computations:

### In Jaxpr:
```
# === BACKWARD (GRADIENT) JAXPR ===
{ lambda ; a:f32[100]. let
    # Forward pass
    b:f32[100] = mul a 2.0
    # Backward pass (reverse mode AD)
    grad_b:f32[100] = broadcast_in_dim ...
    grad_a:f32[100] = mul grad_b 2.0
  in (grad_a,) }
```

### In HLO:
```
// === BACKWARD (GRADIENT) HLO ===
HloModule jit_grad_func, ...

ENTRY main.6 {
  Arg_0.1 = f32[100]{0} parameter(0)
  // ... gradient computation ...
}
```

## Optimized HLO

After XLA compilation, includes:
- Operation fusion (combining multiple ops)
- Memory layout optimizations
- Platform-specific optimizations
- Constant folding

## Getting IR Programmatically

```python
import jax
import jax.numpy as jnp

def func(a, b):
    return a * 2.0 + b

sample_a = jnp.ones((100,))
sample_b = jnp.ones((100,))

# Jaxpr
jaxpr = jax.make_jaxpr(func)(sample_a, sample_b)
print(jaxpr)

# HLO (before optimization)
lowered = jax.jit(func).lower(sample_a, sample_b)
print(lowered.as_text())

# Optimized HLO (after XLA compilation)
compiled = lowered.compile()
print(compiled.as_text())
```
