# JAX IR Formats

## 1. Jaxpr (JAX Expression)
Jaxpr is a simplified, functional intermediate representation specific to JAX.
- **Structure**:
  - `lambda`: Input arguments with types.
  - `let`: Sequence of assignments (primitive operations).
  - `in`: Return values.
- **Types**: Typed arrays (e.g., `f32[3,4]`), tuples.
- **Primitives**: `add`, `mul`, `dot_general`, `scan`, `cond`, etc.

## 2. StableHLO (MLIR)
When lowered, JAX produces StableHLO, which is a dialect of MLIR (Multi-Level Intermediate Representation).
- **Structure**:
  - `module`: Top-level container.
  - `func.func`: Functions. `@main` is the entry point.
- **Ops**: Prefixed with `stablehlo.` (e.g., `stablehlo.add`, `stablehlo.constant`).
- **Tensors**: Typed tensors (e.g., `tensor<f32>`, `tensor<10x20xf32>`).
- **Attributes**: Metadata like `dense<...>` for constants.

This is the representation fed into the XLA compiler backend.
