# StableHLO IR Format Reference

## Module Structure
```
module @jit_<fn_name> attributes {...} {
  func.func public @main(...) -> (... {jax.result_info = "..."}) {
    // operations
    return %result : type
  }
}
```

## Common Operations
| StableHLO Op | Description |
|-------------|-------------|
| `stablehlo.add` | Element-wise addition |
| `stablehlo.multiply` | Element-wise multiplication |
| `stablehlo.dot_general` | Matrix/tensor contraction |
| `stablehlo.sine`, `stablehlo.cosine` | Trigonometric ops |
| `stablehlo.reduce` | Reduction (sum, max, etc.) |
| `stablehlo.broadcast_in_dim` | Broadcasting scalars/tensors |
| `stablehlo.case` | Conditional branching |
| `stablehlo.while` | Loop construct (from scan) |

## Type Notation
- Scalars: `tensor<f32>`, `tensor<i32>`
- Vectors: `tensor<NxT>` e.g., `tensor<3xf32>`
- Matrices: `tensor<MxNxT>` e.g., `tensor<8x4xf32>`
- Boolean: `tensor<i1>`

## Key Patterns
- Constants: `%cst = stablehlo.constant dense<value> : type`
- Function calls: `func.call @fn_name(...) : (...) -> ...`
- Returns: `return %val : type` or `stablehlo.return %val : type`
