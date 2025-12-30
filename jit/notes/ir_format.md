# IR Format Documentation

## Jaxpr Format
```
{ lambda ; <inputs>. let
    <var>:<type> = <primitive>[<params>] <args>
    ...
  in (<outputs>,) }
```

### Key Components
- `lambda`: Closure variables (usually empty)
- Inputs: Typed variables like `a:f32[4,4]`
- Primitives: `add`, `mul`, `dot_general`, `reduce_sum`, etc.
- Outputs: Tuple of result variables

## StableHLO Format (MLIR)
```mlir
module @jit_<name> attributes {...} {
  func.func public @main(<args>) -> (<result>) {
    %0 = stablehlo.<op> %args : <types>
    return %n : <type>
  }
}
```

### Key Operations
- `stablehlo.add`, `stablehlo.multiply`
- `stablehlo.dot_general` with dimension specs
- `stablehlo.reduce` for reductions
- `stablehlo.case` for conditionals
