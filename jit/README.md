# Warp JIT Code Synthesis - CUDA Backend

This project synthesizes Python→IR paired data from NVIDIA Warp kernels for LLM training.
Supports both CPU and CUDA backends.

**Key Insight**: CUDA IR code generation works WITHOUT a GPU! Only kernel execution requires GPU hardware. This means you can generate large-scale CUDA training datasets on any machine.

## Requirements

```bash
pip install warp-lang
```

For CUDA support, you need:
- NVIDIA GPU with CUDA support
- CUDA drivers installed
- `nvidia-smi` should show your GPU

## Quick Start

### CPU Mode (default)

```bash
# Generate 100 CPU IR samples
python3 code/synthesis/pipeline.py -n 100 -o data/cpu

# Large-scale CPU generation
python3 code/synthesis/batch_generator.py -n 10000 -o data/large
```

### CUDA Mode (No GPU Required for Generation!)

```bash
# Generate CUDA IR samples (works on any machine!)
python3 code/synthesis/pipeline.py -n 100 --device cuda -o data/cuda

# Large-scale CUDA production (recommended)
python3 code/synthesis/cuda_pipeline.py -n 10000 -o data/cuda_training

# Validate generated data
python3 code/synthesis/validate_cuda_data.py data/cuda_training

# Alternative: batch generator
python3 code/synthesis/batch_generator.py -n 10000 --device cuda -o data/cuda_large
```

### CUDA Execution Tests (Requires GPU)

```bash
# Check CUDA availability
python3 -c "import warp as wp; wp.init(); print('CUDA:', wp.is_cuda_available())"

# Run CUDA test suite (requires GPU)
python3 tests/test_cuda.py

# Or run full test script
bash tests/run_cuda_tests.sh
```

## File Structure

```
jit/
├── code/
│   ├── extraction/           # IR extraction utilities
│   │   ├── ir_extractor.py   # Extract IR from kernels
│   │   └── analyze_cuda_diff.py  # CPU vs CUDA analysis
│   ├── synthesis/            # Data synthesis pipeline
│   │   ├── generator.py      # Kernel generators (6 types)
│   │   ├── pipeline.py       # Single-process pipeline
│   │   ├── batch_generator.py # Optimized batch generation
│   │   ├── cuda_pipeline.py  # Dedicated CUDA production
│   │   └── validate_cuda_data.py # Data validation
│   └── examples/             # Example kernels
├── data/
│   ├── cuda_samples/         # Sample CUDA pairs (50 in git)
│   └── cuda_training/        # Full CUDA dataset (generated locally)
├── tests/
│   ├── test_cuda.py          # CUDA test suite
│   └── run_cuda_tests.sh     # Full test script
└── notes/
    └── cuda_analysis.md      # CPU vs CUDA technical analysis
```

## Generated CUDA Data

Successfully generated 10,000 CUDA Python→IR pairs:

| Metric | Value |
|--------|-------|
| Total pairs | 10,000 |
| Success rate | 100% |
| Generation speed | ~540 pairs/sec |
| GPU required | No (generation only) |

Category distribution:
- arithmetic: 1,696 (17.0%)
- vector: 1,704 (17.0%)
- matrix: 1,669 (16.7%)
- control_flow: 1,671 (16.7%)
- math: 1,667 (16.7%)
- atomic: 1,593 (15.9%)

## Kernel Types

The generator supports 6 kernel categories:

| Category | Description | Example Operations |
|----------|-------------|-------------------|
| arithmetic | Basic math | add, sub, mul, div, min, max |
| math | Unary functions | sin, cos, exp, sqrt, abs |
| control_flow | Conditionals/loops | if/else, for loops |
| vector | Vector operations | dot, cross, length, normalize |
| matrix | Matrix operations | mat-vec, mat-mat, transpose |
| atomic | Atomic operations | atomic_add, atomic_min, atomic_max |

## Generated Data Format

```json
{
  "python_source": "@wp.kernel\ndef kernel_name(...):\n    ...",
  "cpp_forward": "void kernel_name_hash_cuda_kernel_forward(...) { ... }",
  "metadata": {
    "kernel_name": "...",
    "category": "arithmetic|math|control_flow|vector|matrix|atomic",
    "description": "...",
    "device": "cpu|cuda"
  }
}
```

## CPU vs CUDA IR Differences

Key differences in generated code:

| Aspect | CPU | CUDA |
|--------|-----|------|
| Function suffix | `_cpu_kernel_forward` | `_cuda_kernel_forward` |
| Thread model | Sequential per task | Grid-stride loop |
| Thread indexing | `builtin_tid1d()` | `blockDim.x * blockIdx.x + threadIdx.x` |
| Shared memory | None | `wp::tile_shared_storage_t` |

Both backends use the same `wp::` namespace functions (wp::add, wp::load, etc.).

## CLI Options

### pipeline.py

```
-n, --count     Number of pairs to generate (default: 100)
-o, --output    Output directory (default: data/samples)
-s, --seed      Random seed (default: 42)
-c, --categories  Specific categories to generate
-d, --device    Target device: cpu or cuda (default: cpu)
```

### batch_generator.py

```
-n              Number of pairs (default: 10000)
-o, --output    Output directory (default: data/large)
-s, --seed      Random seed (default: 42)
-d, --device    Target device: cpu or cuda (default: cpu)
```

## Development Notes

- CUDA IR can be generated without a GPU (for code analysis)
- Actual execution requires a CUDA-capable GPU
- Run `tests/test_cuda.py` on a GPU machine to verify functionality
