# Warp JIT Code Synthesis - CUDA Backend

This project synthesizes Python→IR paired data from NVIDIA Warp kernels for LLM training.
Supports both CPU and CUDA backends.

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

### CUDA Mode

```bash
# Check CUDA availability
python3 -c "import warp as wp; wp.init(); print('CUDA:', wp.is_cuda_available())"

# Generate CUDA IR samples
python3 code/synthesis/pipeline.py -n 100 --device cuda -o data/cuda

# Large-scale CUDA generation
python3 code/synthesis/batch_generator.py -n 10000 --device cuda -o data/cuda_large

# Run CUDA test suite
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
│   │   └── batch_generator.py # Optimized batch generation
│   └── examples/             # Example kernels
├── data/
│   ├── cpu/                  # CPU-generated samples
│   └── cuda/                 # CUDA-generated samples
├── tests/
│   ├── test_cuda.py          # CUDA test suite
│   └── run_cuda_tests.sh     # Full test script
└── notes/
    └── cuda_analysis.md      # CPU vs CUDA technical analysis
```

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
