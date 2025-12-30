# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → XLA HLO/MHLO (with forward and backward passes).

## Overview

This project uses JAX's JIT compilation to generate high-quality Python→XLA HLO/MHLO training pairs for large language models. Each sample contains:
- Python function source code
- **XLA HLO code** (High Level Optimizer intermediate representation)
- **Optimized HLO** with compiler optimization passes applied
- **MHLO code** (MLIR HLO dialect representation)
- **Forward and backward functions** (gradient computation)

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "scalar_arr_qahf",
  "python": "def scalar_arr_qahf(...):\n    ...",
  "hlo": "HLO module IR...",
  "optimized_hlo": "Optimized HLO IR...",
  "mhlo": "MHLO module IR...",
  "type": "generate_scalar_array_op"
}
```

Each sample includes:
- **`hlo`**: XLA HLO text representation (includes forward + backward)
- **`optimized_hlo`**: Optimized HLO after compiler passes
- **`mhlo`**: MLIR-based HLO representation (if available)

## Kernel Types (10 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `return a + b` |
| `scalar_array` | Scalar + array ops | `return alpha * x + y` |
| `unary` | Math functions | `return jnp.sin(a)` |
| `branch` | Conditionals | `jnp.where(a > 0, ...)` |
| `loop` | For loops | `jax.lax.fori_loop(...)` |
| `reduction` | Reduction ops | `jnp.sum(a)` |
| `vector` | Vector operations | `jnp.dot(a, b)` |
| `multi_statement` | Chained ops | `temp = a+b; return jnp.sqrt(temp)` |
| `nested_branch` | Nested conditions | `jnp.where(a > 0, jnp.where(...))` |
| `compound` | Mixed patterns | Complex multi-op functions |

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
# or
pip install 'jax[cpu]>=0.4.23' numpy
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file with all IR representations
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl --include-mhlo

# Generate without MHLO
python3 code/synthesis/pipeline.py --count 1000 --output data/hlo_only.jsonl --jsonl
```

### Test IR Extraction
```bash
python3 jit/code/extraction/ir_extractor.py
```

## Project Structure

```
jit/
├── code/
│   ├── extraction/
│   │   └── ir_extractor.py      # Core IR extraction (HLO + MHLO)
│   └── synthesis/
│       ├── generator.py         # 10 kernel type generators
│       ├── pipeline.py          # Main synthesis pipeline
│       └── batch_generator.py   # Scalable batch generation
├── data/
│   ├── training_all.jsonl       # Main dataset
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── jax_basics.md            # JAX compilation flow
    └── ir_format.md             # HLO IR structure docs
```

## How It Works

1. **Kernel Generation**: `generator.py` creates random Python functions from 10 templates
2. **JIT Compilation**: JAX compiles functions to XLA HLO
3. **IR Extraction**: `ir_extractor.py` captures the generated HLO/MHLO code
4. **Gradient Computation**: Automatic differentiation creates backward pass
5. **Pair Creation**: Pipeline combines Python + HLO + MHLO into training samples

## Key Features

- **HLO + MHLO**: Both intermediate representations included
- **Forward + Backward**: Both gradient functions included (critical for differentiable programming)
- **Optimized IR**: Includes compiler-optimized versions
- **10 Kernel Types**: Balanced coverage of common computation patterns
- **Reproducible**: Seeded random generation for reproducibility
- **Production Ready**: Validated, clean JSONL format

## JAX vs Warp

This project was migrated from NVIDIA Warp to JAX:

| Aspect | Warp | JAX |
|--------|------|-----|
| Target | C++/CUDA | XLA HLO |
| Backend | CPU/CUDA | CPU/GPU/TPU |
| IR Format | C++ source | HLO/MHLO |
| Decorator | `@wp.kernel` | `@jax.jit` |
| Arrays | `wp.array` | `jnp.array` |
| Auto-diff | Built-in | `jax.grad` |

### Why JAX?

- **Portable**: Runs on CPU, GPU, TPU without code changes
- **Functional**: Pure functional programming model
- **Composable**: Easy to combine with other JAX transformations
- **HLO IR**: Industry-standard compiler IR used by XLA
- **Ecosystem**: Large ML/AI community and libraries

## XLA HLO Format

XLA HLO (High Level Optimizer) is the intermediate representation used by:
- JAX
- TensorFlow
- PyTorch (via torch.compile)
- Other ML frameworks

Example HLO for `a + b`:
```
HloModule jit_add

ENTRY main {
  p0 = f32[8] parameter(0)
  p1 = f32[8] parameter(1)
  ROOT add = f32[8] add(p0, p1)
}
```

## License

MIT License. JAX is licensed under Apache 2.0.
