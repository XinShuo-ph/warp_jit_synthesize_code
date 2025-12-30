# JAX JIT Compilation Flow

## 1. Tracing (Python -> Jaxpr)
When `jax.jit` or `jax.make_jaxpr` is called, JAX runs the Python function with tracer objects.
This generates a "jaxpr" (JAX expression), which is a data-flow graph of JAX primitives.
- **Jaxpr**: Typed, functional, intermediate representation.

## 2. Lowering (Jaxpr -> StableHLO / MLIR)
The Jaxpr is then lowered into StableHLO (a dialect of MLIR) using `jit(fn).lower(...)`.
StableHLO is the input to the XLA compiler. It is device-agnostic but preserves high-level semantics.

## 3. Compilation (StableHLO -> Machine Code)
XLA (Accelerated Linear Algebra) takes the StableHLO and optimizes it (fusion, layout optimizations).
Finally, it emits machine code (PTX for GPU, AVX for CPU).

## Key Functions
- `jax.make_jaxpr(f)(*args)`: View the Jaxpr.
- `jax.jit(f).lower(*args)`: Lower to `Lowered` stage (contains HLO/MLIR).
- `lowered.as_text()`: Get HLO text.
- `lowered.compile()`: Compile to executable.
