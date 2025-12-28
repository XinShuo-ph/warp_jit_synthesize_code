# Warp JIT Code Synthesis - instructions-wrapup-completion-981e

## Progress Summary
- **Milestone reached**: M5 (Complete)
- **Key deliverables**:
  - IR extraction system for Warp kernels (CPU + CUDA)
  - Random kernel generator with 7 strategies
  - Synthesis pipeline for creating python→IR pairs
  - Batch generation system with multiprocessing
  - 10,000 training samples in JSONL format
  - 100 individual sample files in JSON format

## What Works
- **IR Extraction**: Extracts C++ (CPU) and CUDA C++ (GPU) source code from `@wp.kernel` decorated functions using `warp.codegen.codegen_kernel()`
- **Kernel Generation**: Generates random kernels with 7 patterns: elementwise, conditional, loop, vec3_op, atomic_accumulate, nested_loop, complex_math
- **Synthesis Pipeline**: Dynamically generates kernels, saves to temp files, loads via `importlib`, extracts IR, and saves pairs
- **Batch Generation**: Parallel generation using multiprocessing with spawn context (safe for CUDA/Warp)
- **Data Output**: Both JSONL (bulk) and individual JSON sample files

## Requirements

```bash
pip install warp-lang
```

## Quick Start

```bash
# Test IR extraction
python3 jit/code/extraction/ir_extractor.py

# Generate 100 samples
python3 jit/code/synthesis/pipeline.py

# Generate large dataset (10k samples, parallel)
python3 jit/code/synthesis/batch_generator.py

# Run IR extraction tests
python3 jit/code/examples/test_ir_extraction.py
```

## File Structure

```
jit/
├── code/
│   ├── extraction/
│   │   └── ir_extractor.py    # Core IR extraction from Warp kernels
│   ├── synthesis/
│   │   ├── generator.py       # Random kernel code generator
│   │   ├── pipeline.py        # Single-threaded synthesis pipeline
│   │   ├── batch_generator.py # Multi-process batch generation
│   │   └── compute_stats.py   # Dataset statistics computation
│   └── examples/
│       ├── test_ir_extraction.py  # IR extraction tests (5 patterns)
│       ├── test_generation.py     # Generator tests (deprecated)
│       ├── poisson_solver.py      # Example Warp kernel
│       └── test_poisson.py        # Poisson solver tests
├── data/
│   ├── samples/               # 100 individual JSON sample files
│   └── large_dataset/
│       └── dataset.jsonl      # 10,000 samples in JSONL format
├── notes/
│   ├── warp_basics.md         # Warp compilation flow overview
│   ├── ir_format.md           # IR structure documentation
│   └── data_stats.md          # Dataset statistics
├── tasks/                     # Milestone task definitions
├── STATE.md                   # Project state tracker
├── WRAPUP_STATE.md            # Wrapup progress tracker
└── instructions.md            # Original project instructions
```

## Generated Data Format

```json
{
  "id": 0,
  "kernel_name": "kernel_0",
  "python_code": "@wp.kernel\ndef kernel_0(in1: wp.array(dtype=int), ...):\n    tid = wp.tid()\n    ...",
  "ir_code": "extern \"C\" __global__ void kernel_0_hash_cuda_kernel_forward(\n    wp::launch_bounds_t dim,\n    wp::array_t<wp::int32> var_in1,\n    ...\n) {\n    ...\n}"
}
```

## Dataset Statistics
- **Total Samples**: 10,000
- **Average IR Lines**: 48.4
- **Strategy Distribution**:
  - elementwise: 28.0%
  - nested_loop: 14.7%
  - vec3_op: 14.5%
  - loop: 14.4%
  - conditional: 13.7%
  - atomic_accumulate: 13.6%
  - complex_math: 1.1%

## Known Issues / TODOs
- `test_generation.py` uses deprecated `exec()` approach (Warp requires file-based loading via `importlib`)
- GPU/CUDA testing not performed (environment has no GPU)
- IR extraction generates CUDA IR even without GPU (Warp's codegen doesn't require GPU for code generation)
- Some temp modules accumulate in `temp_modules/` directory during pipeline runs
