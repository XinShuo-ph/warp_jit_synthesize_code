# XLA HLO Format Reference

## Structure
```
HloModule <name>, entry_computation_layout={...}

<helper_computations>   # Optional sub-functions (relu, reduce regions, etc.)

ENTRY main.X {
  <param> = <type> parameter(N)
  <op> = <type> <operation>(<operands>), <attributes>
  ROOT <result> = <type> <final_op>(...)
}
```

## Common Operations
- Elementwise: `add`, `subtract`, `multiply`, `divide`, `maximum`, `minimum`, `negate`
- Unary: `exp`, `log`, `sqrt`, `tanh`, `abs`
- Matrix: `dot`, `dot_general` (with contraction dims)
- Shape: `reshape`, `transpose`, `broadcast`, `slice`
- Reduce: `reduce` with region (sum, max, etc.)
- Control: `conditional`, `while`, `call`

## Type Format
- `f32[4,8]{1,0}` = float32 tensor, shape [4,8], layout {1,0} (row-major)
- `s32[]` = scalar int32
- Batch dims in dot: `lhs_batch_dims={0}, rhs_batch_dims={0}`

## Key Observations
- Functions are inlined or defined as regions
- `ROOT` marks the return value
- Broadcasts are explicit (unlike numpy)
