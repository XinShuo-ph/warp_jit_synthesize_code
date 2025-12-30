# JAX Compilation Flow

## Python → XLA HLO Pipeline

1. **Function Definition**: Regular Python function with JAX operations
2. **JIT Compilation**: `jax.jit()` triggers XLA compilation
3. **Tracing**: JAX traces the function with abstract values
4. **Lowering**: Converts traced computation to XLA HLO
5. **Optimization**: XLA applies optimization passes
6. **Backend Compilation**: XLA compiles to target (CPU/GPU/TPU)
7. **Caching**: Results cached for reuse

## Key Components

- **JAX Tracer**: Tracks operations during function execution
- **XLA Compiler**: Compiles HLO to machine code
- **JIT Cache**: Stores compiled functions for reuse
- **Auto-differentiation**: `jax.grad` generates backward pass

## IR Representations

### HLO (High Level Optimizer)
- Text-based intermediate representation
- Contains all operations and data flow
- Used for optimization passes

### MHLO (MLIR HLO)
- MLIR dialect for HLO
- More structured representation
- Better for advanced transformations

### Optimized HLO
- HLO after XLA optimization passes
- Shows fusion, constant folding, etc.
- Closer to actual execution

## Programmatic IR Access

```python
import jax

# Define function
def my_func(x):
    return x * 2.0 + 1.0

# JIT compile
jitted = jax.jit(my_func)

# Lower to HLO
lowered = jitted.lower(x_example)

# Get HLO text
hlo_text = lowered.as_text()

# Compile and get optimized HLO
compiled = lowered.compile()
optimized_hlo = compiled.as_text()

# Get MHLO representation
mhlo_module = lowered.compiler_ir(dialect='mhlo')
```

## Forward/Backward Code

JAX auto-generates backward (gradient) code via automatic differentiation:

```python
# Forward function
def forward(x):
    return jnp.sum(x ** 2)

# Backward function (automatic)
backward = jax.grad(forward)

# Both forward and backward in single function
def combined(x):
    return forward(x), backward(x)
```

When lowered to HLO, this creates a single module with both computations.

## JAX vs TensorFlow vs PyTorch

All three frameworks use XLA HLO:
- **JAX**: Native XLA compilation via `jax.jit`
- **TensorFlow**: XLA via `tf.function(jit_compile=True)`
- **PyTorch**: XLA via `torch.compile` or `torch_xla`

## Cache Location

JAX compilation cache is stored in:
- `~/.cache/jax/` (Linux)
- Hash includes: function code, input shapes/dtypes, compilation options

## Differentiation Modes

JAX supports multiple differentiation modes:
- **Forward-mode**: `jax.jvp` (Jacobian-vector product)
- **Reverse-mode**: `jax.grad` (most common, efficient for many→few)
- **Vector-Jacobian product**: `jax.vjp`

All generate HLO that includes both primal and derivative computations.
