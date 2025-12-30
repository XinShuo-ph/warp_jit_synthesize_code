# JAX JIT Code Synthesis Dataset

Training data generation pipeline for LLM code translation: Python → XLA HLO (with forward and backward passes).

## Overview

This project uses JAX's JIT compilation via XLA to generate high-quality Python→HLO training pairs for large language models. Each sample contains:
- Python function source code
- **HLO (High Level Optimizer)** intermediate representation
- **Optimized HLO** after XLA optimization passes

## Dataset

**Location:** `jit/data/jax_training_all.jsonl`  
**Format:** JSONL (one JSON per line)

### Sample Format

```json
{
  "id": 0,
  "kernel_name": "elementwise_qahf",
  "python": "@jax.jit\ndef elementwise_qahf(a, b):\n    return a + b",
  "hlo": "HloModule jit_elementwise_qahf...",
  "hlo_optimized": "HloModule jit_elementwise_qahf, optimizations...",
  "type": "generate_simple_elementwise"
}
```

Each sample includes:
- **`python`**: Full Python source with JAX decorators
- **`hlo`**: Unoptimized HLO representation
- **`hlo_optimized`**: XLA-optimized HLO representation

## Function Types (15 categories)

| Type | Description | Example |
|------|-------------|---------|
| `elementwise` | Basic arithmetic (+, -, *) | `return a + b` |
| `scalar_array` | Scalar + array ops | `return alpha * x + y` |
| `unary` | Math functions | `return jnp.sin(a)` |
| `branch` | Conditionals (jnp.where) | `jnp.where(a > 0, ...)` |
| `loop` | jax.lax.fori_loop | `jax.lax.fori_loop(...)` |
| `reduction` | Sum/mean operations | `jnp.sum(a)` |
| `vector` | Dot/norm operations | `jnp.dot(a, b)` |
| `multi_statement` | Chained ops | `temp = a+b; jnp.sqrt(temp)` |
| `nested_branch` | Nested jnp.where | Nested conditionals |
| `compound` | Mixed patterns | Complex multi-op functions |
| `matmul` | Matrix multiplication | `jnp.matmul(a, b)` |
| `softmax` | Softmax activation | Stable softmax impl |
| `scan` | jax.lax.scan | Sequential operations |
| `vmap` | Vectorized mapping | `jax.vmap(...)` |
| `grad` | Gradient computation | `jax.grad(...)` |

## Quick Start

### Requirements
```bash
pip install jax jaxlib
# For GPU support:
# pip install jax[cuda12] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Generate Training Data
```bash
cd jit

# Generate 100 pairs (demo)
python3 code/synthesis/jax_pipeline.py --count 100

# Generate to JSONL file with HLO
python3 code/synthesis/jax_pipeline.py --count 1000 --output data/my_data.jsonl --jsonl

# Generate HLO only (no optimized)
python3 code/synthesis/jax_pipeline.py --count 1000 --output data/hlo_only.jsonl --jsonl --output-type hlo

# Generate both HLO and optimized HLO
python3 code/synthesis/jax_pipeline.py --count 1000 --output data/full.jsonl --jsonl --output-type both
```

### Batch Generation (with checkpointing)
```bash
# Sequential generation with checkpointing
python3 code/synthesis/jax_batch_generator.py --count 1000 --output data/training.jsonl

# Parallel generation (faster)
python3 code/synthesis/jax_batch_generator.py --count 1000 --output data/training.jsonl --parallel
```

### Test IR Extraction
```bash
python3 jit/code/extraction/jax_ir_extractor.py
```

## Project Structure

```
jit/
├── code/
│   ├── extraction/
│   │   ├── ir_extractor.py          # Warp IR extraction (legacy)
│   │   ├── jax_ir_extractor.py      # JAX HLO extraction
│   │   └── test_ir_extractor.py     # Tests
│   └── synthesis/
│       ├── generator.py             # Warp generators (legacy)
│       ├── jax_generator.py         # JAX function generators
│       ├── pipeline.py              # Warp pipeline (legacy)
│       ├── jax_pipeline.py          # JAX synthesis pipeline
│       ├── batch_generator.py       # Warp batch gen (legacy)
│       └── jax_batch_generator.py   # JAX batch generation
├── data/
│   ├── jax_training_all.jsonl       # Main JAX dataset
│   ├── training_all.jsonl           # Warp dataset (legacy)
│   └── samples/
└── notes/
    ├── warp_basics.md               # Warp compilation flow (legacy)
    ├── ir_format.md                 # Warp C++ IR structure (legacy)
    └── jax_hlo_format.md            # JAX HLO structure docs
```

## How It Works

1. **Function Generation**: `jax_generator.py` creates random Python functions from 15 templates
2. **JIT Compilation**: JAX compiles functions via XLA
3. **HLO Extraction**: `jax_ir_extractor.py` captures the HLO representation
4. **Pair Creation**: Pipeline combines Python + HLO into training samples

## Key Features

- **XLA Backend**: Uses Google's XLA compiler for HLO generation
- **Forward + Backward**: Gradient functions can be extracted via `jax.grad`
- **Reproducible**: Seeded random generation for reproducibility
- **15 Function Types**: Comprehensive coverage of JAX patterns
- **Production Ready**: Validated, clean JSONL format
- **GPU/TPU Ready**: JAX supports accelerator backends

## JAX vs Warp

| Feature | JAX | Warp |
|---------|-----|------|
| IR Type | XLA HLO | C++/CUDA |
| Backend | XLA (CPU/GPU/TPU) | CPU/CUDA |
| Autodiff | `jax.grad`, `jax.vjp` | Built-in adjoint |
| Parallelism | `vmap`, `pmap` | Kernel threads |
| Use Case | ML/Scientific | Physics simulation |

## HLO Format

HLO (High Level Optimizer) is XLA's intermediate representation:

```
HloModule jit_my_function

ENTRY main.5 {
  Arg_0.1 = f32[16]{0} parameter(0)
  Arg_1.2 = f32[16]{0} parameter(1)
  ROOT add.3 = f32[16]{0} add(Arg_0.1, Arg_1.2)
}
```

Key elements:
- `HloModule`: Module name
- `parameter(N)`: Input arguments
- Operations: `add`, `multiply`, `reduce`, etc.
- `ROOT`: Output of the computation

## License

Uses JAX (Apache 2.0 license).
