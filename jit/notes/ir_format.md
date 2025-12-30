# JAX IR Format

## JAXPR Structure
```
{ lambda ; <inputs>. let <body> in (<outputs>,) }
```

### Components
- **Inputs**: `varname:dtype[shape]` e.g., `a:f32[4]`, `b:i32[2,2]`
- **Body**: Sequence of `varname:dtype[shape] = primitive[params] args`
- **Outputs**: Tuple of output variables

## Common Primitives
| Primitive | Description |
|-----------|-------------|
| `add`, `sub`, `mul`, `div` | Element-wise arithmetic |
| `sin`, `cos`, `exp`, `log` | Element-wise math |
| `reduce_sum`, `reduce_max` | Reductions with `axes` param |
| `dot_general` | Generalized matrix multiply |
| `transpose` | Reorder dimensions |
| `broadcast_in_dim` | Broadcasting |
| `select_n` | Conditional selection (where) |
| `scan` | Sequential loop |

## XLA HLO Structure
```
module @jit_funcname { func.func public @main(...) { ... } }
```
Uses StableHLO ops: `stablehlo.add`, `stablehlo.multiply`, etc.
