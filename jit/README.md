# Warp JIT Code Synthesis

> Merged from 16 branches - production-ready Python→IR training data generation

## Progress Summary
- **Milestone reached**: M5 (All milestones complete)
- **Key deliverables**:
  - IR Extractor: Extract Python→C++ pairs from Warp kernels
  - Poisson Solver: FEM-based solver with validation tests
  - Synthesis Pipeline: Programmatic kernel generation
  - Batch Generator: Large-scale dataset generation (10,500+ pairs)
  - Documentation: Compilation flow, IR format, dataset stats

## What Works
- **IR Extraction**: Extracts Python source and generated C++ code from any `@wp.kernel`
- **7 kernel types validated**: arithmetic, vector ops, matrix ops, control flow, loops, math functions, atomics
- **Poisson Solver**: FEM implementation with 4 validation tests (convergence, boundary conditions, consistency, analytical comparison)
- **Synthesis Pipeline**: End-to-end generation of Python→IR pairs with 7 categories
- **Batch Generation**: ~180 pairs/second throughput, 10,500 pairs generated

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Test IR extraction (7 kernel types)
python3 code/extraction/test_ir_extractor.py

# Generate 10 Python→IR pairs
python3 code/synthesis/pipeline.py -n 10 -o data/test

# Run Poisson solver validation
python3 code/examples/test_poisson.py

# Generate large batch
python3 code/synthesis/batch_generator.py --count 1000 --output data/custom
```

## File Structure

```
jit/
├── code/
│   ├── extraction/           # Python→IR extraction
│   │   ├── ir_extractor.py   # Core extraction logic
│   │   ├── test_ir_extractor.py  # 7 kernel validation tests
│   │   └── save_sample_pairs.py  # Save pairs to JSON
│   ├── synthesis/            # Kernel generation
│   │   ├── generator.py      # Programmatic kernel templates
│   │   ├── pipeline.py       # End-to-end synthesis
│   │   └── batch_generator.py    # Large-scale generation
│   └── examples/             # Example kernels
│       ├── poisson_solver.py     # FEM Poisson solver
│       ├── test_poisson.py       # Solver validation
│       └── test_basic_kernels.py # Basic kernel examples
├── data/
│   ├── samples/              # 125 sample pairs (manual + synthesized)
│   └── large/                # 10,500 pairs (42MB)
└── notes/
    ├── warp_basics.md        # Warp compilation flow docs
    ├── ir_format.md          # Generated IR structure docs
    └── data_stats.md         # Dataset statistics
```

## Generated Data Format

```json
{
  "python_source": "@wp.kernel\ndef kernel_add(a: wp.array(dtype=float), ...):\n    tid = wp.tid()\n    c[tid] = a[tid] + b[tid]\n",
  "cpp_forward": "void kernel_add_..._cpu_kernel_forward(...) {\n    // Generated C++ code\n}",
  "metadata": {
    "kernel_name": "kernel_add",
    "category": "arithmetic",
    "description": "Element-wise addition",
    "device": "cpu"
  }
}
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
| combined     | Multi-pattern kernels              | loop + conditional + math   |

## API Reference

### IR Extraction

```python
from code.extraction.ir_extractor import extract_ir, extract_python_ir_pair

# Full extraction
result = extract_ir(kernel, device="cpu", include_backward=True)
# Returns: python_source, cpp_code, forward_code, backward_code, metadata

# Simple extraction
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
from code.synthesis.pipeline import synthesize_pair, synthesize_batch, run_pipeline

# Single pair
pair = synthesize_pair(spec, device="cpu")

# Batch synthesis
pairs = synthesize_batch(n=100, seed=42)

# Full pipeline with file output
run_pipeline(n=100, output_dir="data/output", seed=42)
```

## Known Issues / TODOs

- **CPU-only**: Current implementation generates CPU C++ code only (no CUDA)
- **ir_extractor.py device param**: Has `device` parameter but CUDA path untested (no GPU available)
- **No GPU validation**: CUDA code generation path not validated
- **Large dataset in git**: 10,500 pairs (42MB) may be large for some workflows
