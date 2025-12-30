# JAX HLO (XLA High Level Optimizer) Format

## Overview

JAX compiles Python functions to XLA's HLO (High Level Optimizer) intermediate representation. HLO is a graph-based IR that represents computations as a directed acyclic graph (DAG) of operations.

## Structure

Generated HLO consists of:
1. **HloModule**: Top-level container with module name
2. **Computations**: Named function-like blocks (ENTRY is the main one)
3. **Instructions**: Operations with typed inputs/outputs
4. **Metadata**: Shape information, debug info

## Basic HLO Layout

```hlo
HloModule jit_my_function, entry_computation_layout={(f32[16]{0}, f32[16]{0})->(f32[16]{0})}

ENTRY main.5 {
  Arg_0.1 = f32[16]{0} parameter(0), metadata={...}
  Arg_1.2 = f32[16]{0} parameter(1), metadata={...}
  ROOT add.3 = f32[16]{0} add(Arg_0.1, Arg_1.2), metadata={...}
}
```

## Key Components

### 1. Module Header
```hlo
HloModule jit_function_name, entry_computation_layout={...}
```
- Module name (derived from Python function)
- Entry computation layout: input/output types

### 2. Parameters
```hlo
Arg_0.1 = f32[16]{0} parameter(0)
```
- `Arg_0.1`: Variable name
- `f32[16]{0}`: Type (float32 array of size 16, layout 0)
- `parameter(0)`: First input argument

### 3. Operations
```hlo
add.3 = f32[16]{0} add(Arg_0.1, Arg_1.2)
multiply.4 = f32[16]{0} multiply(add.3, constant.2)
```
- Operation name with unique suffix
- Result type
- Operation (add, multiply, etc.)
- Input operands

### 4. ROOT
```hlo
ROOT result.10 = f32[16]{0} ...
```
The `ROOT` instruction marks the output of the computation.

## Common Operations

### Arithmetic
| Python | HLO |
|--------|-----|
| `a + b` | `add(a, b)` |
| `a - b` | `subtract(a, b)` |
| `a * b` | `multiply(a, b)` |
| `a / b` | `divide(a, b)` |

### Math Functions
| Python | HLO |
|--------|-----|
| `jnp.sqrt(x)` | `sqrt(x)` |
| `jnp.sin(x)` | `sine(x)` |
| `jnp.cos(x)` | `cosine(x)` |
| `jnp.exp(x)` | `exponential(x)` |
| `jnp.log(x)` | `log(x)` |
| `jnp.abs(x)` | `abs(x)` |

### Reductions
| Python | HLO |
|--------|-----|
| `jnp.sum(x)` | `reduce(..., add, ...)` |
| `jnp.max(x)` | `reduce(..., maximum, ...)` |
| `jnp.mean(x)` | `reduce + divide` |

### Conditionals
```hlo
// jnp.where(cond, a, b)
select.5 = f32[16]{0} select(compare.4, Arg_0.1, Arg_1.2)
```

### Broadcasting
```hlo
broadcast.3 = f32[16]{0} broadcast(constant.2), dimensions={}
```

## Type System

### Scalar Types
- `f32`: 32-bit float
- `f64`: 64-bit float
- `s32`: 32-bit signed int
- `s64`: 64-bit signed int
- `pred`: boolean/predicate

### Array Types
```
f32[16]{0}      # 1D array of 16 floats
f32[8,8]{1,0}   # 2D array (8x8), row-major layout
f32[2,3,4]{2,1,0}  # 3D array with specified layout
```

## Control Flow

### Loops (from jax.lax.fori_loop)
```hlo
while.10 = (s32[], f32[16]{0}) while((s32[], f32[16]{0}) init.9),
    condition=cond_computation, body=body_computation
```

### Conditionals (from jax.lax.cond)
```hlo
conditional.15 = f32[16]{0} conditional(pred.14, ...), 
    true_computation=..., false_computation=...
```

### Scan (from jax.lax.scan)
```hlo
// Represented as a while loop with tuple state
while.20 = (...) while(...), condition=..., body=scan_body
```

## Gradient Computation

When using `jax.grad`, the HLO includes:
1. Forward computation
2. Backward pass (chain rule)
3. Gradient accumulation

Example pattern:
```hlo
// Forward
multiply.5 = f32[16]{0} multiply(Arg_0.1, Arg_0.1)  // x * x
reduce.10 = f32[] reduce(multiply.5, ...), dimensions={0}, to_apply=add_computation

// Backward (gradient of sum(x^2) = 2*x)
broadcast.15 = f32[16]{0} broadcast(constant.14), dimensions={}
multiply.16 = f32[16]{0} multiply(constant_2, Arg_0.1)  // 2 * x
```

## Optimized HLO

After XLA optimization passes, HLO may include:
- Fused operations: `fusion.1 = ... fusion(...), kind=kLoop`
- Layout optimizations
- Constant folding
- Common subexpression elimination

Example:
```hlo
fusion.5 = f32[16]{0} fusion(Arg_0.1, Arg_1.2), kind=kLoop, 
    calls=fused_computation.4
```

## Comparison with Warp C++

| Aspect | JAX HLO | Warp C++ |
|--------|---------|----------|
| Format | Text/Graph IR | C++ source code |
| Types | XLA types (f32, s32) | C++ types (float, int) |
| Parallelism | Implicit | Explicit (wp.tid()) |
| Gradients | Separate computation | Interleaved forward/backward |
| Target | XLA backends | CPU/CUDA directly |

## Debugging

Use JAX's built-in tools:
```python
# Get HLO text
lowered = jax.jit(func).lower(*args)
print(lowered.as_text())

# Get optimized HLO
compiled = lowered.compile()
print(compiled.as_text())

# Get MLIR/StableHLO
print(lowered.as_text(dialect="stablehlo"))
```
