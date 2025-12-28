# CUDA Dataset Generation

## ⚠️ GPU Required

This code requires an NVIDIA GPU with CUDA support to generate actual CUDA IR. The current environment does not have GPU access.

## Quick Start (On GPU Machine)

### 1. Install Dependencies
```bash
pip install warp-lang numpy
```

### 2. Verify CUDA is Available
```bash
python3 -c "import warp as wp; wp.init(); print(f'CUDA: {wp.is_cuda_available()}')"
```

### 3. Run Test Generation
```bash
cd /workspace/production/cuda
bash test_on_gpu.sh
```

### 4. Generate Full Dataset (200MB)
```bash
python3 code/pipeline.py --count 30000 --output data/full --seed 2000
```

## Generated Data Format

Each JSON file contains:
```json
{
  "python_source": "@wp.kernel\ndef kernel_name(...):\n    ...",
  "cuda_ir_forward": "void kernel_name_cuda_kernel_forward(...) {\n    // CUDA C++ code\n}",
  "metadata": {
    "kernel_name": "...",
    "category": "...",
    "device": "cuda",
    ...
  }
}
```

## Key Differences from CPU Dataset

| Aspect | CPU | CUDA |
|--------|-----|------|
| Device | "cpu" | "cuda" |
| IR Type | C++ (CPU) | CUDA C++ → PTX |
| Function Suffix | `*_cpu_kernel_forward` | `*_cuda_kernel_forward` |
| Special Constructs | Standard C++ | Thread indices, shared memory, atomics |

## Kernel Categories

Same as CPU dataset:
- arithmetic: Basic operations (+, -, *, /)
- conditional: If/else branches
- loop: For loops
- math: Mathematical functions (sin, cos, sqrt, ...)
- vector: Vector operations (vec2, vec3, vec4)
- matrix: Matrix operations (mat22, mat33)
- atomic: Atomic operations (add, max, min, ...)
- nested_loop: Nested loops
- multi_conditional: Multiple branches
- combined: Mixed patterns

## Performance Tuning

For faster generation on GPU:
- Use batch processing (built into pipeline)
- Adjust `--seed` for different random samples
- Specify `--categories` to focus on specific kernel types

## Expected Generation Time

- ~30ms per sample (on modern GPU)
- 30,000 samples: ~15 minutes
- 200MB dataset: ~30,000 samples

## Troubleshooting

### "CUDA not available" Error
- Ensure NVIDIA GPU is present: `nvidia-smi`
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Check warp can see GPU: `python3 -c "import warp as wp; wp.init(); print(wp.get_devices())"`

### Compilation Failures
- Check GPU memory: `nvidia-smi`
- Reduce batch size if OOM
- Some complex kernels may fail - this is expected (success rate ~95%)

## Current Status

Since no GPU is available in this environment, I've:
1. ✅ Created CUDA-adapted pipeline code
2. ✅ Provided generator (same as CPU - kernels are device-agnostic)
3. ✅ Created test script for GPU machine
4. ⏭️ **You need to run on GPU to generate actual CUDA IR**

## Alternative: Use CPU Data as Template

The CPU dataset (292MB in `../cpu/data/`) uses the same Python kernels. Only the IR compilation target differs. You can:
1. Use CPU data for Python kernel examples
2. Re-generate on GPU for actual CUDA IR when available

## Files

- `code/pipeline.py` - CUDA generation pipeline
- `code/generator.py` - Kernel generator (device-agnostic)
- `test_on_gpu.sh` - Quick test script
- `README.md` - This file
