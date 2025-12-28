# GPU Testing Guide

This guide explains how to validate the CUDA code generation on a machine with an NVIDIA GPU.

## Requirements

- NVIDIA GPU with CUDA support (compute capability 6.0+)
- CUDA toolkit installed (version 11.0+)
- Python 3.10+
- warp-lang package

## Quick Setup

```bash
# Install warp
pip install warp-lang

# Verify GPU is detected
python -c "import warp as wp; wp.init(); print([d.alias for d in wp.get_devices()])"
# Expected: ['cpu', 'cuda:0'] or similar
```

## Running Tests

### 1. Quick GPU Test
```bash
cd cuda/tests
python run_gpu_tests.py --quick
```

### 2. Full GPU Test Suite
```bash
cd cuda/tests
python run_gpu_tests.py
```

### 3. CPU-only Tests (no GPU required)
These tests verify code generation without executing on GPU:
```bash
cd cuda/tests
python test_extraction.py
python test_kernels.py
```

## Expected Output (GPU Tests)

```
============================================================
GPU Execution Tests (REQUIRES CUDA GPU)
============================================================
Found GPU: NVIDIA GeForce RTX 3080
✓ test_simple_kernel_execution passed
✓ test_math_kernel_execution passed
✓ test_vector_kernel_execution passed
✓ test_atomic_kernel_execution passed
✓ test_generated_kernel_execution passed
✓ test_backward_execution passed
============================================================
Results: 6 passed, 0 failed
============================================================
```

## Generating CUDA Training Data

```bash
cd cuda/code/synthesis

# Generate 100 CUDA Python→IR pairs
python pipeline.py -n 100 -o ../data/samples -d cuda

# Generate CPU pairs for comparison
python pipeline.py -n 100 -o ../data/cpu_samples -d cpu
```

## Data Format

Each generated JSON file contains:

```json
{
  "kernel_name": "arith_abc123",
  "python_source": "@wp.kernel\ndef arith_abc123(...):\n    ...",
  "cuda_forward": "void arith_abc123_xyz_cuda_kernel_forward(...) {\n    ...\n}",
  "cuda_backward": "void arith_abc123_xyz_cuda_kernel_backward(...) {\n    ...\n}",
  "device": "cuda",
  "category": "arithmetic"
}
```

## Kernel Categories

| Category | Description |
|----------|-------------|
| arithmetic | Basic ops (+, -, *, /) |
| conditional | if/else branches |
| loop | for loops |
| math | sin, cos, exp, log, etc. |
| vector | vec2/vec3/vec4 ops |
| atomic | atomic_add, atomic_min, atomic_max |
| nested | nested loops |
| multi_cond | multiple if/elif/else |
| combined | loops + conditionals |
| scalar_param | scalar function parameters |
| random_math | random expression trees |

## Troubleshooting

### "CUDA driver not found"
- Ensure NVIDIA drivers are installed: `nvidia-smi`
- Ensure CUDA toolkit is installed: `nvcc --version`

### "No CUDA GPU found"
- Check GPU is recognized: `lspci | grep -i nvidia`
- Try reinstalling warp: `pip install --force-reinstall warp-lang`

### Kernel execution errors
- Check CUDA memory: `nvidia-smi`
- Reduce batch size if out of memory

## Performance Notes

- Code generation (no GPU): ~0.1s per kernel
- Kernel compilation (first run): ~1-2s per kernel  
- Kernel execution: varies by complexity
- Backward pass: ~2x forward time
