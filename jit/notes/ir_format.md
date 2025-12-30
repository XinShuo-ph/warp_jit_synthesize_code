# IR Format Documentation

## Jaxpr Format

Jaxpr (JAX Expression) is JAX's high-level functional IR.

### Structure
```
{ lambda ; inputs. let bindings in outputs }
```

### Components
- **inputs**: Typed input variables (e.g., `a:f32[3]`)
- **bindings**: Let-expressions defining intermediate values
- **outputs**: Final return values

### Example
```jaxpr
{ lambda ; a:f32[3] b:f32[3]. let c:f32[3] = add a b in (c,) }
```

### Type Notation
- `f32[3]`: float32 array of shape (3,)
- `f32[2,3]`: float32 array of shape (2,3)
- `f32[]`: scalar float32

## StableHLO Format

StableHLO is an MLIR dialect used by XLA for compilation.

### Structure
- Module wrapper with attributes
- Function definitions (public @main)
- Tensor operations with explicit types

### Example
```mlir
module @jit_add attributes {mhlo.num_partitions = 1 : i32} {
  func.func public @main(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>) 
      -> (tensor<3xf32>) {
    %0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>
    return %0 : tensor<3xf32>
  }
}
```

### Key Operations
- `stablehlo.add`, `stablehlo.subtract`: Arithmetic
- `stablehlo.dot_general`: Matrix multiplication
- `stablehlo.reduce`: Reductions (sum, max, etc.)
- `stablehlo.reshape`: Shape manipulation
- `stablehlo.sine`, `stablehlo.exponential`: Math functions
