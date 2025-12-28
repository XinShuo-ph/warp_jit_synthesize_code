# Warp JIT Code Synthesis

## Overview
Python→IR code synthesis pipeline for LLM training data using NVIDIA Warp.

Generates (Python kernel source, C++ IR code) pairs suitable for training models to understand GPU compilation.

## Progress Summary
- **Merged from**: 16 independent development branches
- **Best-of-breed components** from branches 12c4, 9177, ff72, 82cf, aa30
- **10 kernel types** for maximum diversity
- **Production-ready** pipeline with 100% success rate

## What Works
- **IR Extraction** (`code/extraction/ir_extractor.py`): Extracts C++ intermediate representation from Python Warp kernels
- **Kernel Generator** (`code/synthesis/generator.py`): Creates 10 diverse kernel patterns
- **Pipeline** (`code/synthesis/pipeline.py`): End-to-end sample generation with validation
- **Batch Generation** (`code/synthesis/batch_generator.py`): Scalable generation with checkpointing
- **Poisson Solver** (`code/examples/poisson_solver.py`): Working FEM Poisson equation solver

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Test IR extraction
python3 code/extraction/ir_extractor.py

# Generate 10 training samples
python3 code/synthesis/pipeline.py -n 10 -o data/samples

# Generate at scale
python3 code/synthesis/batch_generator.py --count 1000 --output data/large

# Run Poisson solver tests
python3 code/examples/test_poisson.py
```

See `QUICKSTART.md` for detailed usage examples.

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
│   │   └── test_ir_extractor.py
│   ├── synthesis/            # Kernel generation and pipeline
│   │   ├── generator.py      # 10-type kernel generator
│   │   ├── pipeline.py       # End-to-end generation pipeline
│   │   └── batch_generator.py # Scalable batch generation
│   └── examples/             # Example kernels and FEM solver
│       ├── poisson_solver.py # Poisson equation FEM solver
│       └── test_poisson.py   # FEM solver tests
├── data/                     # Generated samples (JSON pairs)
├── notes/                    # Technical documentation
│   ├── warp_basics.md       # Warp compilation flow
│   ├── ir_format.md         # IR structure documentation
│   └── data_stats.md        # Dataset statistics
├── README.md                 # This file
└── QUICKSTART.md            # Quick start guide
```

## Generated Data Format

```json
{
  "python_source": "@wp.kernel\ndef add_arrays(a: wp.array(dtype=float), ...):\n    tid = wp.tid()\n    c[tid] = a[tid] + b[tid]\n",
  "cpp_forward": "void add_arrays_cpu_kernel_forward(...) {\n    var_0 = builtin_tid1d();\n    var_2 = wp::add(var_a, var_b);\n    ...\n}",
  "metadata": {
    "kernel_name": "add_arrays",
    "category": "arithmetic",
    "description": "Arithmetic kernel with 2 operations",
    "device": "cpu"
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
- CPU-only mode (CUDA requires GPU hardware)
- Generated kernels are synthetic (designed for variety, not realistic programs)
- Samples limited in git; use batch_generator for large datasets

---

**Date**: December 28, 2025  
**Status**: Production Ready
