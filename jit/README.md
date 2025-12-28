# Warp JIT Code Synthesis - cursor/instructions-wrapup-completion-efe6

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
- **Synthesis Pipeline**: End-to-end generation of Python→IR pairs with 6 categories
- **Batch Generation**: ~180 pairs/second throughput, 10,500 pairs generated

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Test IR extraction (7 kernel types)
python3 code/extraction/test_ir_extractor.py

# Generate 10 Python→IR pairs (CPU)
python3 code/synthesis/pipeline.py -n 10 -o data/test --device cpu

# Generate 10 Python→IR pairs (CUDA; run on a GPU machine)
python3 code/synthesis/pipeline.py -n 10 -o data/test_cuda --device cuda

# Run Poisson solver validation
python3 code/examples/test_poisson.py

# Generate batch (CPU)
python3 code/synthesis/batch_generator.py -n 1000 -o data/custom --device cpu

# Generate batch (CUDA; run on a GPU machine)
python3 code/synthesis/batch_generator.py -n 1000 -o data/custom_cuda --device cuda
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
│   └── samples/              # small sample pairs (keep git light)
└── notes/
    ├── warp_basics.md        # Warp compilation flow docs
    ├── ir_format.md          # Generated IR structure docs
    └── data_stats.md         # Dataset statistics
```

## Generated Data Format

```json
{
  "python_source": "@wp.kernel\ndef kernel_add(a: wp.array(dtype=float), ...):\n    tid = wp.tid()\n    c[tid] = a[tid] + b[tid]\n",
  "code_forward": "void kernel_add_..._cpu_kernel_forward(...) {\n    // Generated forward kernel code\n}",
  "code_full": "...\n// Full generated module code\n...",
  "cpp_forward": "void kernel_add_..._cpu_kernel_forward(...) { ... }",
  "cu_forward": "__global__ void kernel_add_..._cuda_kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "kernel_add",
    "category": "arithmetic",
    "description": "Element-wise addition",
    "device": "cpu",
    "code_ext": ".cpp"
  }
}
```

## Kernel Categories

| Category     | Description                        | Examples                    |
|--------------|------------------------------------|-----------------------------|
| arithmetic   | Basic scalar operations            | add, sub, mul, div          |
| vector       | Vector operations                  | dot, cross, normalize       |
| matrix       | Matrix operations                  | mat-vec multiply            |
| control_flow | Conditionals                       | if/else, clamp              |
| math         | Math functions                     | sin, cos, exp, sqrt         |
| atomic       | Atomic operations                  | atomic_add, atomic_max      |

## API Reference

### IR Extraction

```python
from code.extraction.ir_extractor import extract_ir, extract_python_ir_pair

# Full extraction
result = extract_ir(kernel, device="cpu", include_backward=True)
# Returns: python_source, cpp_code, forward_code, backward_code, metadata (incl. code_ext)

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
pairs = synthesize_batch(n=100, seed=42, device="cpu")

# Full pipeline with file output
run_pipeline(n=100, output_dir="data/output", seed=42, device="cpu")
```

## Known Issues / TODOs

- **CUDA requires a GPU machine**: CUDA codegen/extraction is implemented, but CPU-only environments will skip CUDA tests.
- **Schema evolution**: Prefer `code_forward`/`code_full` + `metadata.code_ext`. `cpp_forward` (CPU) and `cu_forward` (CUDA) are provided for compatibility.
- **Dataset size**: Avoid committing large generated datasets; prefer `/tmp` for bulk generation.
