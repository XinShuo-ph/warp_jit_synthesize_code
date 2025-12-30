# JAX Basics

## JAX Compilation Flow

JAX uses XLA (Accelerated Linear Algebra) as its compiler backend.

### Compilation Pipeline
1. **Python Code** → JAX tracer captures operations
2. **Jaxpr** → JAX's intermediate representation (functional)
3. **HLO/StableHLO** → XLA High-Level Optimizer IR
4. **LLVM IR** → Low-level machine code (CPU/GPU)

### Key Decorators
- `@jax.jit` - Just-in-time compilation for performance
- `jax.grad` - Automatic differentiation
- `jax.vmap` - Automatic vectorization

## IR Extraction Methods

### Method 1: Using jax.jit().lower()
```python
lowered = jax.jit(func).lower(*args)
hlo_text = lowered.as_text(dialect='hlo')
```

### Method 2: Using compiler_ir()
```python
lowered = jax.jit(func).lower(*args)
hlo_ir = lowered.compiler_ir(dialect='hlo')  # Returns XlaComputation
hlo_text = hlo_ir.as_hlo_text()  # Get text representation
```

### Method 3: StableHLO (MLIR-based)
```python
stablehlo_text = lowered.compiler_ir(dialect='stablehlo')
# Returns string directly
```

## IR Formats Available

### HLO (High-Level Optimizer)
- XLA's native IR
- More verbose, closer to hardware
- Example: `add.1 = f32[3]{0} add(x.1, y.1)`

### StableHLO
- MLIR-based (Multi-Level IR)
- More portable across versions
- Example: `%0 = stablehlo.add %arg0, %arg1 : tensor<3xf32>`

## Key Findings

1. **StableHLO is preferred** - More stable API, better for training data
2. **Lowered representation** - Contains compilation artifacts before execution
3. **Type information** - IR includes tensor shapes and dtypes
4. **Operations** - Maps well to hardware operations (add, mul, dot, etc.)

## IR Storage Location

- IR is generated on-demand via `lower()` method
- Not stored on disk by default (unlike Warp's .cpp/.cu files)
- Can serialize via protobuf if needed

## Use Cases for Training Data

JAX IR is ideal for:
- Tensor operations (add, mul, matmul, etc.)
- Scientific computing patterns
- Automatic differentiation (grad operations)
- Control flow (while, cond, scan)
- Parallel patterns (vmap, pmap)
