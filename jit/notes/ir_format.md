# JAX IR Format (StableHLO)

## Structure
StableHLO uses MLIR (Multi-Level IR) format:
```
module @jit_function_name {
  func.func public @main(...) -> (...) {
    %0 = stablehlo.operation ...
    return %0
  }
}
```

## Key Components
1. **Module**: Container for the entire computation
2. **Function**: Entry point with input/output types
3. **Operations**: StableHLO operations (add, mul, dot, etc.)
4. **Types**: tensor<shape x dtype>

## Common Operations
- `stablehlo.add`: Element-wise addition
- `stablehlo.multiply`: Element-wise multiplication
- `stablehlo.dot_general`: Matrix multiplication
- `stablehlo.sine`, `stablehlo.cosine`: Trig functions
- `stablehlo.exponential`, `stablehlo.log`: Exponentials
- `stablehlo.tanh`: Hyperbolic tangent
- `stablehlo.select`: Conditional selection
- `stablehlo.reduce`: Reduction operations

## Type Representation
- `tensor<3xf32>`: 1D tensor of 3 float32 values
- `tensor<2x2xf32>`: 2D tensor (2x2 matrix) of float32
- `tensor<f32>`: Scalar float32

## Example
```mlir
func.func public @main(%arg0: tensor<3xf32>) -> (tensor<3xf32>) {
  %0 = stablehlo.sine %arg0 : tensor<3xf32>
  return %0 : tensor<3xf32>
}
```
