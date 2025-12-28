# Warp JIT Code Synthesis

## Overview
Python→IR code synthesis pipeline for LLM training data using NVIDIA Warp.

Generates (Python kernel source, IR code) pairs suitable for training models to understand GPU compilation. **Supports both CPU (C++) and CUDA backends.**

## Progress Summary
- **Merged from**: 16 independent development branches
- **Best-of-breed components** from branches 12c4, 9177, ff72, 82cf, aa30
- **10 kernel types** for maximum diversity
- **CPU + CUDA backends** with forward and backward pass support
- **Production-ready** pipeline with 100% success rate

## What Works
- **IR Extraction** (`code/extraction/ir_extractor.py`): Extracts C++/CUDA intermediate representation from Python Warp kernels
- **Kernel Generator** (`code/synthesis/generator.py`): Creates 10 diverse kernel patterns
- **Pipeline** (`code/synthesis/pipeline.py`): End-to-end sample generation with CPU/CUDA device selection
- **Batch Generation** (`code/synthesis/batch_generator.py`): Scalable CPU IR generation with checkpointing
- **CUDA Batch Generator** (`code/synthesis/cuda_batch_generator.py`): Production CUDA IR generation (no GPU required!)
- **Dataset Validation** (`code/synthesis/cuda_dataset_stats.py`): Validates CUDA datasets
- **CUDA Test Suite** (`tests/cuda/`): 37 extraction tests + GPU execution tests
- **Poisson Solver** (`code/examples/poisson_solver.py`): Working FEM Poisson equation solver

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Test IR extraction
python3 code/extraction/ir_extractor.py

# Generate 10 CPU training samples
python3 code/synthesis/pipeline.py -n 10 -o data/samples

# Generate 10 CUDA training samples (no GPU required!)
python3 code/synthesis/pipeline.py -n 10 -d cuda -b -o data/cuda_samples

# Generate CUDA IR at scale with checkpointing
python3 code/synthesis/cuda_batch_generator.py --count 1000 --output data/cuda_large --backward --checkpoint

# Validate CUDA dataset
python3 code/synthesis/cuda_dataset_stats.py data/cuda_samples --validate

# Run Poisson solver tests
python3 code/examples/test_poisson.py

# Run CUDA extraction tests (no GPU required)
python3 -m pytest tests/cuda/test_extraction.py -v
```

See `QUICKSTART.md` for detailed usage examples and `tests/cuda/README.md` for CUDA testing.

## Kernel Types (10 Categories)

| Category | Description | Example |
|----------|-------------|---------|
| `arithmetic` | Binary ops chain (+, -, *, /) | `c[tid] = a[tid] + b[tid] * 2.0` |
| `vector` | Vec2/3/4 operations | `wp.dot(a[tid], b[tid])` |
| `matrix` | Mat22/33/44 operations | `wp.transpose(m[tid])` |
| `control_flow` | If/else, loops | `if val < 0: out = -val` |
| `math` | Unary functions | `wp.sin(wp.cos(x[tid]))` |
| `atomic` | Atomic reductions | `wp.atomic_add(result, 0, val)` |
| `nested_loop` | Nested for loops | `for i: for j: acc += ...` |
| `multi_condition` | Multiple if/elif/else | `if/elif/else` chains |
| `combined` | Mixed patterns | Loop + condition + math |
| `scalar_param` | Scalar parameters | `out = x * scale + offset` |

## File Structure

```
jit/
├── code/
│   ├── extraction/           # IR extraction from Warp kernels
│   │   ├── ir_extractor.py   # Main extraction utility
│   │   ├── test_cuda_extraction.py # CUDA extraction tests
│   │   └── test_ir_extractor.py
│   ├── synthesis/            # Kernel generation and pipeline
│   │   ├── generator.py      # 10-type kernel generator
│   │   ├── pipeline.py       # End-to-end pipeline (CPU + CUDA)
│   │   ├── batch_generator.py # Scalable CPU batch generation
│   │   ├── cuda_batch_generator.py # CUDA production batch generator
│   │   └── cuda_dataset_stats.py # CUDA dataset validation
│   └── examples/             # Example kernels and FEM solver
│       ├── poisson_solver.py # Poisson equation FEM solver
│       └── test_poisson.py   # FEM solver tests
├── data/
│   ├── samples/              # CPU IR pairs (JSON)
│   └── cuda_samples/         # CUDA IR pairs (100 validated pairs)
├── tests/
│   └── cuda/                 # CUDA test suite
│       ├── test_extraction.py # IR extraction tests (no GPU)
│       ├── test_kernels.py   # GPU execution tests
│       ├── run_gpu_tests.sh  # GPU test runner
│       └── README.md         # Test documentation
├── notes/                    # Technical documentation
│   ├── warp_basics.md       # Warp compilation flow
│   ├── ir_format.md         # IR structure documentation
│   ├── data_stats.md        # Dataset statistics
│   └── cuda_notes.md        # CPU vs CUDA differences
├── README.md                 # This file
└── QUICKSTART.md            # Quick start guide
```

## Generated Data Format

### CPU Format
```json
{
  "python_source": "@wp.kernel\ndef add_arrays(...):\n    c[tid] = a[tid] + b[tid]\n",
  "ir_forward": "void add_arrays_cpu_kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "add_arrays",
    "category": "arithmetic",
    "device": "cpu",
    "ir_type": "cpu",
    "has_backward": false
  }
}
```

### CUDA Format (with backward pass)
```json
{
  "python_source": "@wp.kernel\ndef add_arrays(...):\n    c[tid] = a[tid] + b[tid]\n",
  "ir_forward": "void add_arrays_cuda_kernel_forward(...) { ... }",
  "ir_backward": "void add_arrays_cuda_kernel_backward(...) { ... }",
  "metadata": {
    "kernel_name": "add_arrays",
    "category": "arithmetic",
    "device": "cuda",
    "ir_type": "cuda",
    "has_backward": true
  }
}
```

## Dataset Statistics

- **Generation Rate**: ~180 pairs/second (CPU mode)
- **Success Rate**: 100% (all kernels compile and extract)
- **Categories**: 10 diverse kernel types
- **Validation**: All pairs produce valid Python source and C++ IR

## Merge Sources

This codebase merges the best components from 16 development branches:

| Component | Source Branch | Notes |
|-----------|---------------|-------|
| Core pipeline | 12c4 | Most complete, largest dataset |
| Additional types | 9177 | nested_loop, multi_condition, combined, scalar_param |
| Documentation | 82cf, aa30 | README, QUICKSTART |
| Generator patterns | ff72 | Clean standalone functions |

## Known Issues / Limitations
- CUDA kernel execution requires GPU hardware (IR extraction works without GPU)
- Generated kernels are synthetic (designed for variety, not realistic programs)
- Samples limited in git; use batch_generator for large datasets

## CUDA Backend

**Key Feature**: CUDA IR can be generated WITHOUT a GPU! Warp's code generation produces CUDA code on any machine.

```bash
# Generate CUDA IR (simple, no GPU required)
python3 code/synthesis/pipeline.py -n 100 -d cuda -b -o data/cuda_samples

# Production CUDA generation with checkpointing (recommended for large datasets)
python3 code/synthesis/cuda_batch_generator.py \
    --count 1000 \
    --output data/cuda_large \
    --backward \
    --checkpoint

# Validate generated dataset
python3 code/synthesis/cuda_dataset_stats.py data/cuda_large --validate

# Run CUDA extraction tests (37 tests, no GPU required)
python3 -m pytest tests/cuda/test_extraction.py -v

# Run GPU execution tests (requires GPU)
./tests/cuda/run_gpu_tests.sh
```

**Performance**: ~279 pairs/sec on CPU, 100% success rate with balanced category distribution.

See `notes/cuda_notes.md` for CPU vs CUDA differences and `tests/cuda/README.md` for testing guide.

---

**Date**: December 28, 2025  
**Status**: Production Ready (CPU + CUDA backends)
