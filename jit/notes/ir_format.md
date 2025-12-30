# XLA HLO IR Format

## Structure

Generated HLO consists of:
1. **HLO Module**: Top-level container
2. **Computations**: Functions/subgraphs
3. **Instructions**: Individual operations
4. **ENTRY computation**: Main entry point

## HLO Module Layout

```
HloModule module_name

ENTRY computation_name {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ROOT result = f32[8] add(p0, p1)
}
```

## Key Concepts

### Parameters
Input values to the computation:
```
p0 = f32[8] parameter(0)  // First input, array of 8 floats
p1 = f32[1] parameter(1)  // Second input, scalar
```

### Operations
HLO instructions that compute values:
```
add.1 = f32[8] add(p0, p1)
mul.2 = f32[8] multiply(add.1, p0)
```

### ROOT
The output of the computation:
```
ROOT result = f32[8] multiply(mul.2, p1)
```

## Common HLO Operations

| Python/JAX | HLO Instruction |
|------------|-----------------|
| `a + b` | `add(a, b)` |
| `a - b` | `subtract(a, b)` |
| `a * b` | `multiply(a, b)` |
| `a / b` | `divide(a, b)` |
| `jnp.sqrt(a)` | `sqrt(a)` |
| `jnp.sin(a)` | `sine(a)` |
| `jnp.exp(a)` | `exponential(a)` |
| `jnp.sum(a)` | `reduce(a, ...)` |
| `jnp.where(cond, x, y)` | `select(cond, x, y)` |

## Control Flow

### Conditionals
```
%pred = pred[8] compare(p0, p1), direction=GT
%result = f32[8] select(%pred, %true_val, %false_val)
```

### Loops
Represented as `while` loops in HLO:
```
%result = f32[8] while(%init) {
  %body_computation = ...
  %condition_computation = ...
}
```

## Shapes and Types

HLO is strongly typed with explicit shapes:
- `f32[8]` - 1D array of 8 floats
- `f32[8,16]` - 2D array (8x16)
- `f32[]` - Scalar (0D array)
- `pred[8]` - 1D array of booleans

## Backward (Gradient) Code

Gradients appear as separate computations or fused in the same module:

```
HloModule jit_func_with_grad

// Forward computation
forward_computation {
  p0 = f32[8] parameter(0)
  mul = f32[8] multiply(p0, constant(2.0))
  ROOT sum = f32[] reduce(mul, ...)
}

// Backward computation (gradient)
backward_computation {
  grad_out = f32[] parameter(0)
  // Gradient operations...
  ROOT grad_input = f32[8] ...
}

ENTRY main {
  input = f32[8] parameter(0)
  fwd = f32[] call(input), to_apply=forward_computation
  grad = f32[8] call(constant(1.0)), to_apply=backward_computation
  ROOT tuple = (f32[], f32[8]) tuple(fwd, grad)
}
```

## Optimization Passes

XLA applies many optimizations:
- **Algebraic Simplification**: `x * 1` → `x`
- **Constant Folding**: `2 + 3` → `5`
- **Dead Code Elimination**: Remove unused operations
- **Fusion**: Combine operations to reduce memory traffic
- **Layout Optimization**: Choose efficient memory layouts

## MHLO (MLIR HLO)

MHLO is the MLIR dialect for HLO:

```mlir
module attributes {mhlo.num_partitions = 1} {
  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
    %0 = mhlo.constant dense<2.0> : tensor<f32>
    %1 = mhlo.multiply %arg0, %0 : tensor<8xf32>
    return %1 : tensor<8xf32>
  }
}
```

Differences from text HLO:
- Uses SSA (Static Single Assignment) form with `%` values
- Structured with MLIR syntax
- Has type annotations inline
- Better for programmatic manipulation

## Metadata

HLO includes metadata for debugging:
- Source location mapping
- Operation names
- Optimization hints

## Determinism

HLO is deterministic - same input always produces same output (assuming same compilation flags).
