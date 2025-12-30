# HLO (High Level Operations) Format

## Overview

HLO is XLA's intermediate representation. It describes computations in a hardware-agnostic way before being lowered to target-specific code.

## HLO Module Structure

```
HloModule module_name, entry_computation_layout={(shape)->shape}

ENTRY computation_name {
  parameter_0 = type[shape] parameter(0)
  ...
  ROOT result = type[shape] operation(operands)
}
```

## Data Types

| HLO Type | Description |
|----------|-------------|
| `pred` | Boolean |
| `s8`, `s16`, `s32`, `s64` | Signed integers |
| `u8`, `u16`, `u32`, `u64` | Unsigned integers |
| `f16`, `f32`, `f64` | Floating point |
| `bf16` | Brain float 16 |
| `c64`, `c128` | Complex numbers |

## Common Operations

### Elementwise Operations
```
add = f32[10] add(a, b)
multiply = f32[10] multiply(a, b)
subtract = f32[10] subtract(a, b)
divide = f32[10] divide(a, b)
```

### Unary Operations
```
exp = f32[10] exponential(a)
log = f32[10] log(a)
sqrt = f32[10] sqrt(a)
sin = f32[10] sine(a)
cos = f32[10] cosine(a)
tanh = f32[10] tanh(a)
abs = f32[10] abs(a)
negate = f32[10] negate(a)
```

### Reductions
```
reduce = f32[] reduce(input, init), dimensions={0}, to_apply=add_computation
```

### Matrix Operations
```
dot = f32[M,N] dot(a, b), lhs_contracting_dims={1}, rhs_contracting_dims={0}
```

### Broadcasting
```
broadcast = f32[10,20] broadcast(scalar), dimensions={}
```

### Conditionals
```
select = f32[10] select(predicate, true_val, false_val)
```

### Reshaping
```
reshape = f32[5,2] reshape(f32[10] input)
transpose = f32[20,10] transpose(f32[10,20] input), dimensions={1,0}
```

## Fusion

XLA aggressively fuses operations to reduce memory traffic:

```
fusion = f32[10] fusion(a, b), kind=kLoop, calls=fused_computation

fused_computation {
  p0 = f32[10] parameter(0)
  p1 = f32[10] parameter(1)
  add = f32[10] add(p0, p1)
  ROOT mul = f32[10] multiply(add, p0)
}
```

## Forward vs Backward Pass

### Forward Pass Example
```
HloModule forward

ENTRY main {
  x = f32[10] parameter(0)
  two = f32[] constant(2)
  b_two = f32[10] broadcast(two), dimensions={}
  ROOT result = f32[10] multiply(x, b_two)
}
```

### Backward Pass Example (Gradient)
```
HloModule backward

ENTRY main {
  x = f32[10] parameter(0)
  grad_out = f32[10] parameter(1)  # Gradient from upstream
  two = f32[] constant(2)
  b_two = f32[10] broadcast(two), dimensions={}
  ROOT grad_x = f32[10] multiply(grad_out, b_two)
}
```

## Metadata

HLO includes metadata for debugging:
```
add = f32[10] add(a, b), metadata={op_type="Add" op_name="jax.numpy.add"}
```

## Sharding (for distributed)

```
parameter = f32[1024,1024] parameter(0), sharding={devices=[2,1]0,1}
```

## Comparing with Other IRs

| Feature | HLO (XLA) | LLVM IR | MLIR |
|---------|-----------|---------|------|
| Level | High | Low | Multi-level |
| Target | ML workloads | General | Extensible |
| Fusion | Built-in | Manual | Dialect-based |
| Shape info | Explicit | Pointer-based | Configurable |
