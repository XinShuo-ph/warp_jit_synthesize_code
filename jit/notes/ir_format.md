# JAX IR Formats

## Overview

JAX provides multiple intermediate representation (IR) formats through its JIT compilation pipeline:

1. **Jaxpr** - JAX's high-level functional IR
2. **HLO** - XLA's High-Level Optimizer IR
3. **StableHLO** - Portable MLIR-based representation

## Jaxpr (JAX Expression)

### Structure

Jaxpr is a functional IR that represents computations as a sequence of primitive operations:

```
{ lambda ; a:f32[64] b:f32[64]. let
    c:f32[64] = add a b
    d:f32[64] = mul c 2.0
  in (d,) }
```

### Components

- **Lambda bindings**: Input parameters with types (`a:f32[64]`)
- **Let bindings**: Intermediate computations
- **Primitives**: JAX operations (`add`, `mul`, `sin`, etc.)
- **Output tuple**: Return values

### Key Patterns

| Python | Jaxpr Primitive |
|--------|-----------------|
| `a + b` | `add a b` |
| `a * b` | `mul a b` |
| `jnp.sin(x)` | `sin x` |
| `jnp.sum(x)` | `reduce_sum x` |
| `jnp.dot(a, b)` | `dot_general a b` |
| `jnp.where(c, a, b)` | `select c a b` |

## HLO (High-Level Optimizer)

### Structure

HLO is XLA's intermediate representation, closer to hardware:

```
HloModule jit_func, entry_computation_layout={(f32[64]{0}, f32[64]{0})->f32[64]{0}}

ENTRY main {
  p0 = f32[64]{0} parameter(0)
  p1 = f32[64]{0} parameter(1)
  add.0 = f32[64]{0} add(p0, p1)
  constant.0 = f32[] constant(2)
  broadcast.0 = f32[64]{0} broadcast(constant.0), dimensions={}
  ROOT mul.0 = f32[64]{0} multiply(add.0, broadcast.0)
}
```

### Components

- **HloModule**: Top-level container with metadata
- **ENTRY**: Main computation entry point
- **Parameters**: Input tensors with layout info
- **Instructions**: Operations with explicit shapes
- **ROOT**: Output instruction

### Key Patterns

| Operation | HLO Instruction |
|-----------|-----------------|
| Add | `add(a, b)` |
| Multiply | `multiply(a, b)` |
| Reduce | `reduce(input, init, reducer)` |
| Broadcast | `broadcast(scalar), dimensions={}` |
| Dot | `dot(a, b, lhs_contracting, rhs_contracting)` |
| Reshape | `reshape(input)` |
| Transpose | `transpose(input), dimensions={...}` |

## StableHLO (MLIR-based)

### Structure

StableHLO is a portable, versioned dialect built on MLIR:

```mlir
module @jit_func {
  func.func public @main(%arg0: tensor<64xf32>, %arg1: tensor<64xf32>) -> tensor<64xf32> {
    %0 = stablehlo.add %arg0, %arg1 : tensor<64xf32>
    %1 = stablehlo.constant dense<2.0> : tensor<f32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %3 = stablehlo.multiply %0, %2 : tensor<64xf32>
    return %3 : tensor<64xf32>
  }
}
```

### Benefits

- Versioned and stable API
- MLIR infrastructure compatibility
- Cross-platform portability
- Better tooling support

## Gradient (Backward) Code

JAX's autodiff generates gradient code automatically:

### Forward Pass
```python
def func(a, b):
    return a + b
```

### Backward Pass (via jax.grad)
```python
grad_func = jax.grad(lambda a, b: jnp.sum(func(a, b)))
```

### Backward Jaxpr
```
{ lambda ; a:f32[64] b:f32[64]. let
    _:f32[64] = add a b           # Forward (unused)
    c:f32[] = broadcast_in_dim[...] 1.0  # Gradient seed
    d:f32[64] = broadcast_in_dim c        # Expand to input shape
  in (d,) }  # Gradient w.r.t. first input
```

## Extraction Methods

### Get Jaxpr
```python
from jax import make_jaxpr
jaxpr = make_jaxpr(func)(a, b)
print(jaxpr)
```

### Get HLO
```python
import jax
lowered = jax.jit(func).lower(a, b)
print(lowered.as_text())
```

### Get StableHLO
```python
# Method 1: via as_text
print(lowered.as_text(dialect='stablehlo'))

# Method 2: via compiler_ir
print(lowered.compiler_ir(dialect='stablehlo'))
```

## Comparison

| Aspect | Jaxpr | HLO | StableHLO |
|--------|-------|-----|-----------|
| Level | High | Low | Medium |
| Readability | ★★★★★ | ★★★☆☆ | ★★★★☆ |
| Hardware detail | Low | High | Medium |
| Portability | JAX only | XLA | Cross-platform |
| Versioning | None | None | Versioned |

## Use Cases for LLM Training

- **Jaxpr**: Best for teaching high-level patterns
- **HLO**: Best for hardware-aware optimization
- **StableHLO**: Best for cross-platform deployment

All three formats are valuable for training code translation models.
