# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → HLO/XLA (with forward and backward passes).

## Overview

This project uses JAX's JIT compilation and XLA to generate high-quality Python→HLO training pairs for large language models. Each sample contains:
- Python function source code
- **HLO (High-Level Optimizer) IR** with forward and backward functions
- **Optimized HLO** after XLA optimization passes

## Dataset

**Location:** `jit/data/training_all.jsonl`  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "function_name": "scalar_arr_qahf",
  "python": "@jax.jit\ndef scalar_arr_qahf(...):\n    ...",
  "hlo": "HloModule jit_scalar_arr_qahf\n\n...",
  "optimized_hlo": "HloModule jit_scalar_arr_qahf (optimized)\n\n...",
  "type": "generate_scalar_array_op",
  "backend": "cpu"
}
```

Each sample includes:
- **`hlo`**: HLO IR with forward + backward (gradient) functions
- **`optimized_hlo`**: Optimized HLO after XLA passes (optional)
- **`backend`**: Target backend (cpu or gpu)

## Function Types (10 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `return a + b` |
| `scalar_array` | Scalar + array ops | `return alpha * x + y` |
| `unary` | Math functions | `return jnp.sin(a)` |
| `branch` | Conditionals | `return jnp.where(a > 0, ...)` |
| `loop` | For loops (via scan) | `jax.lax.scan(...)` |
| `reduction` | Reduction ops | `return jnp.sum(a)` |
| `vector` | Vector operations | `return jnp.dot(a, b)` |
| `multi_statement` | Chained ops | `temp = a+b; return sqrt(temp)` |
| `nested_branch` | Nested conditionals | `jnp.where(..., jnp.where(...))` |
| `compound` | Mixed patterns | Complex multi-op functions |

## Quick Start

### Requirements
```bash
pip install jax[cuda12] numpy  # For GPU support
# or
pip install jax[cpu] numpy     # For CPU only
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs (demo)
python3 code/synthesis/pipeline.py --count 100

# Generate to JSONL file with optimized HLO
python3 code/synthesis/pipeline.py --count 1000 --output data/my_data.jsonl --jsonl --include-optimized

# Generate CPU-targeted data
python3 code/synthesis/pipeline.py --count 1000 --output data/cpu_data.jsonl --jsonl --backend cpu

# Generate GPU-targeted data (requires GPU)
python3 code/synthesis/pipeline.py --count 1000 --output data/gpu_data.jsonl --jsonl --backend gpu
```

### Test IR Extraction
```bash
python3 jit/code/extraction/ir_extractor.py
```

### Test Examples
```bash
python3 jit/code/examples/test_add_kernel.py
python3 jit/code/examples/test_dot_product.py
python3 jit/code/examples/test_saxpy.py
```

## Project Structure

```
jit/
├── code/
│   ├── extraction/
│   │   └── ir_extractor.py      # Core HLO/XLA IR extraction
│   ├── synthesis/
│   │   ├── generator.py         # 10 function type generators
│   │   ├── pipeline.py          # Main synthesis pipeline
│   │   └── batch_generator.py   # Scalable batch generation
│   └── examples/
│       ├── test_add_kernel.py   # Simple addition example
│       ├── test_dot_product.py  # Dot product example
│       └── test_saxpy.py        # SAXPY operation example
├── data/
│   ├── training_all.jsonl       # Main dataset
│   └── samples/                 # Sample pairs (JSON)
└── notes/
    ├── ir_format.md             # HLO IR structure docs
    └── warp_basics.md           # (legacy - Warp docs)
```

## How It Works

1. **Function Generation**: `generator.py` creates random Python functions from 10 templates
2. **JIT Compilation**: JAX compiles functions using XLA for CPU/GPU backends
3. **IR Extraction**: `ir_extractor.py` captures the generated HLO IR
4. **Pair Creation**: Pipeline combines Python + HLO into training samples

## Key Features

- **HLO IR**: Direct access to XLA's high-level intermediate representation
- **Forward + Backward**: Both value and gradient computations included
- **XLA Optimization**: Both unoptimized and optimized HLO available
- **Reproducible**: Seeded random generation for reproducibility
- **10 Function Types**: Balanced coverage of common numerical patterns
- **Multi-Backend**: Support for CPU and GPU targets

## HLO IR Advantages

**HLO (High-Level Optimizer)** is XLA's IR that:
- Represents computation graphs explicitly
- Shows fusion, parallelization, and memory optimizations
- Is hardware-agnostic (can target CPU, GPU, TPU)
- Includes gradient computation (via XLA autodiff)

## CPU vs GPU Compilation

**CPU backend** generates:
- Sequential or vectorized operations
- CPU-specific optimizations

**GPU backend** generates:
- Parallel kernel launches
- GPU-specific optimizations (fusion, shared memory)
- CUDA/ROCm-specific patterns

## Backward Pass (Gradient Computation)

JAX automatically generates gradient code via autodifferentiation:

```python
@jax.jit
def forward(x):
    return x * 2.0

# Gradient is automatically computed
grad_fn = jax.grad(forward)
```

The HLO IR includes both forward and backward computations in the same module.

## License

Uses JAX (Apache-2.0 license).
