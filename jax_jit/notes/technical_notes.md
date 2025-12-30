# JAX JIT Technical Notes

## StableHLO IR Format

StableHLO is the intermediate representation used by JAX. It's a dialect of MLIR (Multi-Level Intermediate Representation).

### Key Characteristics

1. **Type System**: Strongly typed tensors with explicit shapes
   ```mlir
   tensor<4xf32>      # 1D tensor of 4 float32s
   tensor<2x3xf32>    # 2x3 matrix
   tensor<f32>        # Scalar (0-dimensional tensor)
   ```

2. **SSA Form**: Static Single Assignment - each value defined once
   ```mlir
   %0 = stablehlo.add %arg0, %arg1
   %1 = stablehlo.multiply %0, %arg2
   ```

3. **Functional**: No side effects, pure transformations

### Compilation Flow

```
Python JAX Code
    ↓
JAX Tracer (JIT)
    ↓
StableHLO IR
    ↓
XLA Compiler
    ↓
Target Code (CPU/GPU/TPU)
```

## Cost Analysis

JAX provides built-in cost analysis:

```python
lowered = jax.jit(func).lower(*args)
cost = lowered.cost_analysis()
```

Returns:
- `flops`: Floating point operations
- `transcendentals`: Special functions (sin, exp, etc.)
- `bytes accessed`: Memory traffic
- `bytes accessed{N}{}`: Per-input memory
- `utilization{N}{}`: Reuse factor

## IR Extraction Methods

### Method 1: Direct Lowering (Used in this project)

```python
lowered = jax.jit(func).lower(*args)
ir_module = lowered.compiler_ir(dialect='stablehlo')
ir_text = str(ir_module)
```

**Pros**: Simple, clean, direct access
**Cons**: Requires example inputs

### Method 2: Using jax.make_jaxpr

```python
jaxpr = jax.make_jaxpr(func)(*args)
```

**Pros**: Shows JAX's intermediate format (Jaxpr)
**Cons**: Not StableHLO, requires further lowering

### Method 3: Compilation

```python
compiled = jax.jit(func).lower(*args).compile()
# Access backend-specific code
```

**Pros**: Can access final compiled code
**Cons**: Backend-specific, harder to parse

## Generator Design

### Shape Compatibility Strategy

To avoid shape mismatch errors:

1. **Same Shape Markers**: Use `array_same` descriptor
   - All `array_same` parameters get identical shapes
   
2. **Matrix Compatibility**: Use `matrix_compatible` descriptor
   - Ensures A.shape[1] == B.shape[0] for matmul

3. **Shape Validation**: Generate inputs after function spec

### Randomization Strategy

- **Function names**: Random 6-letter suffix
- **Constants**: Uniform random in appropriate ranges
- **Operations**: Random selection from curated lists
- **Shapes**: Random selection from predefined safe shapes

## Performance Optimization

### Generation Speed

Achieved ~136 pairs/sec through:

1. **Batch Processing**: Generate 100 pairs at once
2. **Minimal Validation**: Only essential checks during generation
3. **Efficient JSON**: Direct serialization without pretty-printing during batch
4. **Shape Caching**: Reuse compatible shapes within batch

### Memory Efficiency

- Stream processing: Don't load all pairs into memory
- JSON storage: Efficient on-disk format
- Lazy compilation: Only compile when extracting IR

## Validation Strategy

### Three-Level Validation

1. **Syntax Check**: JSON parseable, required fields present
2. **Compilation Check**: Function compiles and lowers
3. **Execution Check**: Function runs with example inputs

### Common Failure Modes

1. **Shape Mismatches**: Incompatible broadcast shapes
   - **Fix**: Use shape-aware generation

2. **Division by Zero**: Random constants include 0
   - **Fix**: Exclude 0 from divisor generation

3. **Domain Errors**: sqrt(-1), log(-1)
   - **Fix**: Ensure positive inputs for domain-restricted functions

4. **Type Errors**: Mixing int/float incorrectly
   - **Fix**: Consistent type usage in generated code

## Dataset Quality Metrics

### Coverage Metrics

- **Operation Coverage**: % of StableHLO ops represented
- **Category Balance**: Distribution across 7 categories
- **Complexity Distribution**: Range of IR line counts

### Diversity Metrics

- **Uniqueness**: % of unique Python/IR pairs
- **Pattern Variety**: Number of distinct operation sequences
- **Shape Variety**: Distribution of tensor shapes

### Quality Metrics

- **Compilation Rate**: % that successfully compile
- **Execution Rate**: % that successfully execute
- **IR Validity**: % with well-formed StableHLO

## Current Dataset Characteristics

### Operation Distribution

Most common StableHLO operations:
1. `broadcast_in_dim` (83%): Scalar to tensor broadcasting
2. `multiply` (61%): Arithmetic multiplication
3. `add` (27%): Arithmetic addition
4. `dot_general` (14%): Matrix operations
5. `reduce` (8%): Aggregations

### Complexity Distribution

- **Simple** (3-8 IR lines): 35%
- **Medium** (9-15 IR lines): 55%
- **Complex** (16+ IR lines): 10%

### FLOPs Distribution

- **Low** (<16 FLOPs): 20%
- **Medium** (16-64 FLOPs): 60%
- **High** (>64 FLOPs): 20%

## Extending the Generator

### Adding New Categories

1. Create generator method in `generator.py`:
```python
def gen_new_category(self) -> FunctionSpec:
    # Define parameters, body, etc.
    pass
```

2. Register in `category_generators` dict

3. Add to default categories list

### Adding New Operations

1. Add to operation lists (e.g., `UNARY_FUNCS`)
2. Ensure JAX support
3. Add validation rules if needed

### Custom Input Generation

Modify `generate_example_inputs()` for new descriptor types:

```python
elif "custom_type" in desc.lower():
    # Generate appropriate input
    inputs.append(custom_generator())
```

## Debugging Tips

### IR Inspection

```python
# Print full IR
lowered = jax.jit(func).lower(*args)
print(lowered.as_text())

# Check compilation
compiled = lowered.compile()
print(compiled.runtime_executable())
```

### Trace Execution

```python
# Use JAX's debug tools
with jax.log_compiles():
    result = jitted_func(*args)
```

### Validate Single Pair

```python
from code.synthesis.pipeline import SynthesisPipeline

pipeline = SynthesisPipeline()
pair = pipeline.generate_single(category='arithmetic')
is_valid = pipeline.validate_pair(pair)
print(f"Valid: {is_valid}")
```
