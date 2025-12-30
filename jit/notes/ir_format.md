# JAX/XLA IR Format (StableHLO)

JAX uses XLA (Accelerated Linear Algebra) as its backend compiler. When we extract the "IR" from a JAX function, we typically get **StableHLO** (Stable High Level Optimizer) dialect of MLIR (Multi-Level Intermediate Representation).

## Structure
- **Module**: The top-level container.
- **Functions**: `func.func @name(...)`.
- **Operations**: `stablehlo.add`, `stablehlo.dot_general`, `stablehlo.sine`, etc.
- **Types**: `tensor<10xf32>` (10-element float32 tensor), `tensor<f32>` (scalar).

## Example
```mlir
module @jit_simple_kernel {
  func.func public @main(%arg0: tensor<10xf32>, %arg1: tensor<10xf32>) -> tensor<10xf32> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<10xf32>
    %1 = stablehlo.sine %arg0 : tensor<10xf32>
    %2 = stablehlo.add %0, %1 : tensor<10xf32>
    return %2 : tensor<10xf32>
  }
}
```

This format is suitable for training LLMs to reason about tensor operations and compilation logic.
