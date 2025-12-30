# JAX JIT Code Synthesis for LLM Training Data

## Overview

This project generates Python→StableHLO paired data for training Large Language Models on JIT compilation. Unlike the original Warp-based implementation, this uses JAX (Google's accelerated linear algebra library) to extract StableHLO intermediate representations.

**Key Achievement**: Production-ready pipeline that generates high-quality Python→IR pairs at **~136 pairs/second**.

## Architecture

### Core Components

```
jax_jit/
├── code/
│   ├── extraction/          # IR extraction from JAX functions
│   │   ├── ir_extractor.py      # Extract StableHLO from JAX
│   │   └── test_ir_extractor.py # Comprehensive tests
│   ├── synthesis/           # Data generation pipeline
│   │   ├── generator.py         # Function generator (7 categories)
│   │   ├── pipeline.py          # End-to-end pipeline
│   │   ├── batch_generator.py   # High-throughput batch generation
│   │   └── validate_dataset.py  # Validation & analysis tools
│   └── examples/
│       └── explore_jax_ir.py    # JAX compilation exploration
├── data/
│   └── samples/             # Generated training pairs (JSON)
└── notes/                   # Documentation
```

## Key Differences from Warp Implementation

| Aspect | Warp Version | JAX Version |
|--------|-------------|-------------|
| **IR Format** | C++ (CUDA/PTX) | StableHLO (MLIR) |
| **Execution Model** | CUDA kernels with `wp.tid()` | Functional array operations |
| **Backend** | GPU-focused | CPU/GPU/TPU portable |
| **Compilation** | Custom kernel compiler | XLA compiler |
| **IR Extraction** | Parse cached C++ files | Direct MLIR access via `lowered.compiler_ir()` |
| **Generation Speed** | ~50-80 pairs/sec | ~136 pairs/sec |

## Quick Start

### 1. Generate Training Data

```bash
# Generate 1000 pairs
python3 code/synthesis/batch_generator.py --count 1000 --batch-size 100

# Generate balanced dataset (equal categories)
python3 code/synthesis/batch_generator.py --count 700 --balanced

# Custom output directory
python3 code/synthesis/batch_generator.py --count 500 --output-dir ./custom_data
```

### 2. Validate Dataset

```bash
python3 code/synthesis/validate_dataset.py
```

### 3. Extract IR from Custom Function

```python
import jax.numpy as jnp
from code.extraction.ir_extractor import extract_ir

def my_function(x, y):
    return jnp.dot(x, y) + 1.0

x = jnp.array([1., 2., 3.])
y = jnp.array([4., 5., 6.])

pair = extract_ir(my_function, x, y)
print(pair.python_source)
print(pair.stablehlo_ir)
print(pair.cost_analysis)
```

## Function Categories

The generator produces 7 categories of functions:

1. **Arithmetic**: Binary operations with constants
   - Example: `(x + y) * 2.5`

2. **Conditional**: Branching logic with `jnp.where`
   - Example: `jnp.where(x > 0, x * 2, x * 0.5)`

3. **Reduction**: Aggregation operations
   - Example: `jnp.sum(x * x) * scale`

4. **Matrix**: Linear algebra operations
   - Example: `jnp.matmul(A, B)`, `jnp.dot(x, y)`

5. **Elementwise**: Unary math functions
   - Example: `jnp.sin(x * 2.0) * 0.5`

6. **Broadcasting**: Scalar-vector operations
   - Example: `scalar * vector + constant`

7. **Composite**: Multi-step computations
   - Example: `jnp.sin(x) + 1.5` combined with `jnp.cos(y) * 2.0`

## StableHLO IR Format

JAX compiles to StableHLO, a stable MLIR dialect:

```mlir
module @jit_saxpy attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<f32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> (tensor<3xf32>) {
    %0 = stablehlo.convert %arg0 : tensor<f32>
    %1 = stablehlo.broadcast_in_dim %0, dims = [] : (tensor<f32>) -> tensor<3xf32>
    %2 = stablehlo.multiply %1, %arg1 : tensor<3xf32>
    %3 = stablehlo.add %2, %arg2 : tensor<3xf32>
    return %3 : tensor<3xf32>
  }
}
```

### Common StableHLO Operations

- **Arithmetic**: `stablehlo.add`, `stablehlo.multiply`, `stablehlo.divide`
- **Shape**: `stablehlo.broadcast_in_dim`, `stablehlo.reshape`
- **Comparison**: `stablehlo.compare GT/LT/EQ`
- **Control**: `stablehlo.select` (conditional)
- **Linear Algebra**: `stablehlo.dot_general`
- **Reduction**: `stablehlo.reduce`

## Dataset Statistics

Current dataset (500 pairs):

```
Total pairs: 500
Generation time: 3.67s
Rate: 136.17 pairs/sec

Categories:
  mixed: 500

Python code:
  Avg lines: 3.3
  Range: 3-5 lines

StableHLO IR:
  Avg lines: 11.8
  Range: 6-19 lines

FLOPs:
  Total: 21,733
  Avg: 43.5 per function
  Range: 8-128
```

## Validation Results

- **Valid pairs**: 99%+
- **Unique pairs**: 100% (no duplicates)
- **Executable**: All functions compile and execute correctly
- **IR coverage**: 10+ unique StableHLO operations

## Advanced Usage

### Extract Forward and Backward Pass

```python
from code.extraction.ir_extractor import extract_with_grad

def loss_fn(x):
    return jnp.sum(x * x)

x = jnp.array([1., 2., 3.])
fwd_pair, bwd_pair = extract_with_grad(loss_fn, x)

print(f"Forward: {len(fwd_pair.stablehlo_ir.splitlines())} IR lines")
print(f"Backward: {len(bwd_pair.stablehlo_ir.splitlines())} IR lines")
```

### Programmatic Pipeline

```python
from code.synthesis.pipeline import SynthesisPipeline

pipeline = SynthesisPipeline(output_dir="./my_data", seed=42)

# Generate single pair
pair = pipeline.generate_single(category='matrix', save=True)

# Generate batch
pairs = pipeline.generate_batch(count=100, save=True, verbose=True)

# Get statistics
stats = pipeline.get_statistics()
print(f"Generated {stats['total_pairs']} pairs")
```

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Generation rate | ~136 pairs/sec |
| Avg Python code size | 3.3 lines |
| Avg IR size | 11.8 lines |
| Avg compilation time | ~7ms per function |
| Memory usage | <100MB for 1000 pairs |
| Validation success | 99%+ |

## Comparison with Original Warp Implementation

### Advantages of JAX Version

1. **Portability**: Runs on CPU/GPU/TPU without code changes
2. **IR Stability**: StableHLO is a stable, versioned IR format
3. **Performance**: 70% faster generation (136 vs 80 pairs/sec)
4. **Simplicity**: No kernel cache management needed
5. **Ecosystem**: Integrates with broader ML ecosystem (TensorFlow, PyTorch via IREE)
6. **Debugging**: Better error messages and debugging tools

### Trade-offs

1. **IR Verbosity**: StableHLO is more verbose than optimized CUDA C++
2. **Low-level Control**: Less direct hardware control than Warp kernels
3. **Specialization**: Warp IR better for CUDA-specific features (atomics, shared memory)

## Use Cases

This dataset is suitable for training LLMs on:

1. **Code→IR Translation**: Teaching models to predict compiler output
2. **Optimization Learning**: Understanding compiler optimizations
3. **Performance Prediction**: Estimating FLOPs/memory from source
4. **Semantic Understanding**: Learning operation equivalences
5. **Cross-language Learning**: Mapping high-level ops to low-level IR

## Future Extensions

- [ ] Add vmap/pmap support for batched operations
- [ ] Include control flow primitives (scan, while_loop)
- [ ] Generate custom gradient functions
- [ ] Add dynamic shapes support
- [ ] Include XLA optimization passes
- [ ] Generate multi-device/distributed patterns

## References

- **JAX**: https://github.com/google/jax
- **StableHLO**: https://github.com/openxla/stablehlo
- **XLA**: https://www.tensorflow.org/xla
- **MLIR**: https://mlir.llvm.org/
