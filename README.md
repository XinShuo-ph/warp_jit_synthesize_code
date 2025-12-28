# Warp JIT Code Synthesis

> **Production-ready Python→IR training data generation for NVIDIA Warp kernels**

Merged from 16 parallel development branches, this project provides a complete pipeline for generating high-quality Python→C++ IR pairs for training Large Language Models on GPU kernel code generation.

## Overview

This project automatically generates training data pairs consisting of:
- **Python source**: Warp kernel functions with `@wp.kernel` decorator
- **C++ IR**: Generated intermediate representation from Warp's JIT compiler
- **Metadata**: Kernel category, argument types, and generation parameters

**Dataset Statistics**: 100 sample pairs included (branches generated 10,000-10,500+ pairs total)

## Quick Start

### Installation

```bash
pip install warp-lang
```

### Generate Your First Samples

```bash
# Generate 10 Python→IR pairs
python3 code/synthesis/pipeline.py --count 10 --output data/my_samples

# Validate generated dataset
python3 code/synthesis/validate_dataset.py data/my_samples

# Analyze dataset statistics
python3 code/synthesis/analyze_dataset.py data/my_samples
```

### Test the Pipeline

```bash
# Test IR extraction (7 kernel types)
python3 code/extraction/test_ir_extractor.py

# Run Poisson FEM solver validation
python3 code/examples/test_poisson.py

# Check installation
python3 code/examples/check_install.py
```

## Features

### IR Extraction
- ✓ Extracts Python source and C++ IR from any `@wp.kernel`
- ✓ Supports CPU and CUDA devices
- ✓ Includes forward and backward (adjoint) kernels
- ✓ 7 kernel types validated: arithmetic, vector, matrix, control flow, loops, math functions, atomics

### Kernel Generation
- ✓ 6 core kernel categories with randomized variations
- ✓ Template-based generation for consistent quality
- ✓ Configurable complexity and patterns
- ✓ Seed support for reproducible generation

### Synthesis Pipeline
- ✓ End-to-end generation: kernel → compile → extract → save
- ✓ Batch generation: ~180 pairs/second throughput
- ✓ JSON output format with full metadata
- ✓ Error handling and validation

### Validation & Analysis
- ✓ Dataset validation utility
- ✓ Statistical analysis tools
- ✓ Categorized test cases
- ✓ Quality assurance suite

### Examples & Documentation
- ✓ Poisson FEM solver with validation tests
- ✓ Classic HPC kernels (add, saxpy, reduction)
- ✓ IR exploration utilities
- ✓ Comprehensive documentation (warp_basics.md, ir_format.md)

## File Structure

```
workspace/
├── code/
│   ├── extraction/           # IR extraction from Warp kernels
│   │   ├── ir_extractor.py   # Core extraction logic
│   │   ├── test_ir_extractor.py  # 7 kernel validation tests
│   │   └── save_sample_pairs.py  # Save pairs to JSON
│   ├── synthesis/            # Kernel generation and pipeline
│   │   ├── generator.py      # 6 kernel category templates
│   │   ├── pipeline.py       # End-to-end synthesis
│   │   ├── batch_generator.py    # Large-scale generation
│   │   ├── validate_dataset.py   # Dataset validation
│   │   └── analyze_dataset.py    # Statistical analysis
│   └── examples/             # Example kernels and utilities
│       ├── poisson_solver.py     # FEM Poisson solver
│       ├── test_poisson.py       # Solver validation
│       ├── ex_add.py         # Classic add kernel
│       ├── ex_saxpy.py       # SAXPY kernel
│       ├── ex_reduction.py   # Reduction kernel
│       ├── explore_ir.py     # IR exploration utility
│       └── check_install.py  # Installation checker
├── data/                     # Generated training samples (100 included)
├── tests/
│   └── cases/                # Categorized test cases
│       ├── case_arith.py     # Arithmetic tests
│       ├── case_atomic.py    # Atomic operation tests
│       ├── case_branch.py    # Branching tests
│       ├── case_loop.py      # Loop tests
│       └── case_vec.py       # Vector operation tests
├── notes/
│   ├── warp_basics.md        # Warp compilation flow
│   ├── ir_format.md          # IR structure documentation
│   └── data_stats.md         # Dataset statistics
├── README.md                 # This file
├── QUICKSTART.md             # Quick start guide
└── FINAL_REPORT.md           # Project completion report
```

## Kernel Categories

| Category     | Description                        | Examples                    |
|--------------|------------------------------------|-----------------------------|
| arithmetic   | Basic scalar operations            | add, sub, mul, div          |
| vector       | Vector operations                  | dot, cross, normalize       |
| matrix       | Matrix operations                  | mat-vec multiply            |
| control_flow | Conditionals and loops             | if/else, clamp, for loops   |
| math         | Math functions                     | sin, cos, exp, sqrt         |
| atomic       | Atomic operations                  | atomic_add, atomic_max      |

## Generated Data Format

Each sample is a JSON file with:

```json
{
  "python_source": "@wp.kernel\ndef kernel_add(a: wp.array(dtype=float), ...):\n    tid = wp.tid()\n    c[tid] = a[tid] + b[tid]\n",
  "cpp_forward": "void kernel_add_..._cpu_kernel_forward(...) {\n    // Generated C++ code\n}",
  "metadata": {
    "kernel_name": "kernel_add",
    "category": "arithmetic",
    "description": "Element-wise addition",
    "device": "cpu",
    "seed": 42
  }
}
```

## API Reference

### IR Extraction

```python
from code.extraction.ir_extractor import extract_ir, extract_python_ir_pair

# Full extraction (with metadata)
result = extract_ir(kernel, device="cpu", include_backward=True)
# Returns: python_source, cpp_code, forward_code, backward_code, metadata

# Simple extraction (Python + forward IR only)
python_src, cpp_forward = extract_python_ir_pair(kernel, device="cpu")
```

### Kernel Generation

```python
from code.synthesis.generator import generate_kernel, generate_kernels

# Generate single kernel
spec = generate_kernel(category="arithmetic", seed=42)

# Generate batch
specs = generate_kernels(n=100, categories=["arithmetic", "vector"], seed=42)
```

### Synthesis Pipeline

```python
from code.synthesis.pipeline import run_pipeline

# Generate training data
run_pipeline(count=100, output_dir="data/output", categories=None, seed=42)
```

### Batch Generation

```bash
# Generate large dataset with checkpointing
python3 code/synthesis/batch_generator.py --count 10000 --output data/large_batch --resume
```

## Performance

- **Throughput**: ~180 pairs/second (single-threaded on CPU)
- **Quality**: 100% validation pass rate on generated samples
- **Scale**: Tested at 10,000+ pairs per run
- **Diversity**: Randomized parameters ensure varied training data

## Validation

```bash
# Validate all samples in a directory
python3 code/synthesis/validate_dataset.py data/my_samples

# Analyze dataset statistics
python3 code/synthesis/analyze_dataset.py data/my_samples
```

## Documentation

- **QUICKSTART.md**: Step-by-step getting started guide
- **FINAL_REPORT.md**: Complete project report
- **notes/warp_basics.md**: Warp kernel compilation flow
- **notes/ir_format.md**: Generated IR structure
- **notes/data_stats.md**: Dataset statistics

## Examples

### Classic HPC Kernels
```bash
python3 code/examples/ex_add.py      # Element-wise addition
python3 code/examples/ex_saxpy.py    # SAXPY operation
python3 code/examples/ex_reduction.py # Reduction sum
```

### FEM Solver
```bash
python3 code/examples/test_poisson.py  # Run all validation tests
```

### IR Exploration
```bash
python3 code/examples/explore_ir.py    # Explore IR structure
```

## Known Limitations

- **CPU-only**: Current samples are CPU-focused (CUDA path untested without GPU)
- **Single-threaded**: Batch generator is single-threaded (future: multiprocessing)
- **No deduplication**: Generated samples not deduplicated across runs

## Contributing

This project was developed through 16 parallel agent branches, each exploring different approaches:
- **Tier 1**: Production-ready implementations (12c4, 9177, 8631)
- **Tier 2**: Feature contributions (82cf, aa30, ff72, 3576, 3a5b)
- **Tier 3-4**: Specialized utilities (25e7, 5d09, a4fd, 0fbe, 7288, 3f34, 4b76, d623)

## License

This project uses NVIDIA Warp, which is licensed under the NVIDIA Source Code License.

## Citation

If you use this dataset or pipeline in your research, please cite NVIDIA Warp:

```
@software{warp2022,
  author = {NVIDIA Corporation},
  title = {Warp: A Python framework for high performance GPU simulation and graphics},
  url = {https://github.com/NVIDIA/warp},
  year = {2022}
}
```
