# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → XLA HLO (with forward and backward passes).

## Overview

This project uses Google JAX's JIT compilation to generate high-quality Python→XLA HLO training pairs for large language models. Each sample contains:
- Python function source code
- **XLA HLO representation** with forward and backward (gradient) functions
- **Optimized HLO** (when available)

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "scalar_arr_qahf",
  "python": "def scalar_arr_qahf(alpha, x, y):\n    \"\"\"Scalar and array operations.\"\"\"\n    return alpha * x + y",
  "hlo": "... HloModule with forward and backward passes ...",
  "optimized_hlo": "... optimized XLA HLO code ...",
  "type": "generate_scalar_array_op"
}
```

Each sample includes:
- **`python`**: Python function source code
- **`hlo`**: XLA HLO intermediate representation with forward + backward passes
- **`optimized_hlo`**: Optimized HLO (when available)

## Kernel Types (10 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `return a + b` |
| `scalar_array` | Scalar + array ops | `return alpha * x + y` |
| `unary` | Math functions | `return jnp.sin(a)` |
| `branch` | Conditionals | `jnp.where(a > 0, ...)` |
| `loop` | Reductions/scans | `jax.lax.scan(...)` |
| `reduction` | Sum/reduce ops | `return jnp.sum(a)` |
| `vector` | Vector operations | `return jnp.sum(a * b, axis=-1)` |
| `multi_statement` | Chained ops | `temp = a+b; return jnp.sqrt(temp)` |
| `nested_branch` | Nested conditionals | `jnp.where(a > 0, jnp.where(...))` |
| `compound` | Mixed patterns | Complex multi-op functions |

## Quick Start

### Requirements
```bash
pip install -r requirements.txt
# or manually:
pip install jax[cpu] jaxlib numpy
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl

# Specify device (note: JAX HLO is device-agnostic until final compilation)
python3 code/synthesis/pipeline.py --count 1000 --output data/training.jsonl --jsonl --device both
```

### Test IR Extraction
```bash
python3 jit/code/extraction/ir_extractor.py
python3 jit/code/extraction/test_ir_extractor.py
```

## Project Structure

```
jit/
├── code/
│   ├── extraction/
│   │   ├── ir_extractor.py      # Core XLA HLO extraction
│   │   └── test_ir_extractor.py # Test IR extraction
│   ├── synthesis/
│   │   ├── generator.py         # 10 kernel type generators
│   │   ├── pipeline.py          # Main synthesis pipeline
│   │   └── batch_generator.py   # Scalable batch generation
│   └── examples/
│       ├── test_add_kernel.py   # Simple addition example
│       ├── test_saxpy.py        # SAXPY example
│       └── test_dot_product.py  # Dot product example
├── data/
│   ├── training_all.jsonl       # Main dataset
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── warp_basics.md           # (legacy) Warp documentation
    └── ir_format.md             # (legacy) C++ IR docs
```

## How It Works

1. **Function Generation**: `generator.py` creates random Python functions from 10 templates
2. **JIT Compilation**: JAX compiles functions to XLA HLO (High-Level Operations)
3. **IR Extraction**: `ir_extractor.py` captures XLA HLO and optimized representations
4. **Pair Creation**: Pipeline combines Python source + XLA HLO into training samples

## Key Features

- **XLA HLO**: Standard intermediate representation used by JAX, TensorFlow, and PyTorch 2.0+
- **Forward + Backward**: Automatic differentiation with gradient functions
- **Device-Agnostic**: XLA HLO compiles to CPU, GPU, TPU
- **Reproducible**: Seeded random generation for reproducibility
- **10 Function Types**: Balanced coverage of common array operations
- **Production Ready**: Validated, clean JSONL format

## XLA HLO Advantages

**XLA HLO** (High-Level Operations):
- Device-agnostic intermediate representation
- Used by JAX, TensorFlow, PyTorch 2.0+
- Optimized by XLA compiler for target hardware
- Includes fusion, layout optimization, memory planning
- Supports automatic differentiation

**JAX Benefits**:
- Pure functional programming model
- Composable transformations (jit, grad, vmap, pmap)
- NumPy-compatible API
- Strong ecosystem and community

## License

Uses Google JAX (Apache 2.0 license).
