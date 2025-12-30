# JAX Basics: JIT Compilation and IR Extraction

## JAX Version
- Version: 0.8.2
- Backend: CPU (default)
- Can extract IR without GPU hardware

## JIT Compilation Flow

1. **Python function** → decorated with `@jax.jit`
2. **Tracing phase** → JAX traces the function with abstract values
3. **Jaxpr generation** → High-level JAX IR (functional, SSA form)
4. **Lowering to StableHLO** → Compiler-level IR (MLIR dialect)
5. **XLA compilation** → Machine code (CPU/GPU/TPU)

## IR Formats

### 1. Jaxpr (JAX Expression)
- High-level functional IR
- Easy to read and understand
- Extraction: `jax.make_jaxpr(fn)(*args)`
- Example: `{ lambda ; a:f32[3] b:f32[3]. let c:f32[3] = add a b in (c,) }`

### 2. StableHLO (formerly HLO)
- Compiler-level IR (MLIR dialect)
- Used by XLA for optimization
- Extraction: `jax.jit(fn).lower(*args).as_text()`
- Format: MLIR text representation
- Contains: module structure, function signatures, tensor operations

## Extraction Methods

### Method 1: Extract Jaxpr
```python
from jax import make_jaxpr
jaxpr = make_jaxpr(function)(*example_inputs)
jaxpr_str = str(jaxpr)  # Convert to string
```

### Method 2: Extract StableHLO
```python
from jax import jit
lowered = jit(function).lower(*example_inputs)
hlo_text = lowered.as_text()  # StableHLO in MLIR format
```

## Key Observations

1. **Jaxpr is simpler**: Good for understanding computation graph
2. **StableHLO is detailed**: Shows low-level tensor operations
3. **Both are deterministic**: Same function → same IR
4. **No GPU needed**: Can extract IR on CPU backend
5. **Shapes matter**: IR depends on input tensor shapes

## What We Can Extract

- Arithmetic operations (add, mul, div, etc.)
- Array operations (broadcast, reshape, slice)
- Linear algebra (matmul, dot)
- Math functions (sin, cos, exp, log)
- Reductions (sum, mean, max, min)
- Control flow (via lax.cond, lax.scan, lax.while_loop)
