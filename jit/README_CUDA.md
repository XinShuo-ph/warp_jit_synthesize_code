# CUDA Backend Development

This directory contains the tools for generating and validating Python-to-CUDA-IR pairs for LLM training.

## Prerequisites

- `warp-lang` installed (`pip install warp-lang`)
- NVIDIA GPU with CUDA drivers (for execution/verification)
- CPU-only environment (for generation/extraction)

## Key Files

- `code/synthesis/pipeline.py`: Main generation script. Now supports `--device`.
- `code/extraction/ir_extractor.py`: Extracts CUDA IR from kernels.
- `cuda_tests/smoke_test.py`: Simple script to verify CUDA environment.
- `cuda_tests/verify_kernels.py`: Validates generated samples by running them on a device.

## Usage

### 1. Check GPU Setup
Run the smoke test to ensure your environment is ready:
```bash
python cuda_tests/smoke_test.py
```

### 2. Generate Data (CPU or GPU)
You can generate CUDA IR even on a CPU-only machine (uses `warp`'s offline codegen):
```bash
# Generate 100 pairs targeting CUDA
python code/synthesis/pipeline.py -n 100 -d cuda --output data/cuda_samples
```

### 3. Verify Generated Kernels (Requires GPU)
Run the generated Python source on the GPU to ensure correctness:
```bash
python cuda_tests/verify_kernels.py data/cuda_samples --device cuda
```

## Generated Data Format

The JSON output now includes:
- `cpp_forward`: The generated CUDA C++ kernel code.
- `metadata.device`: "cuda"
- `metadata.arg_types`: Type information for reconstruction/verification.

Example:
```json
{
  "python_source": "@wp.kernel...",
  "cpp_forward": "extern \"C\" __global__ void ...",
  "metadata": {
    "device": "cuda",
    "arg_types": {"a": "wp.array(dtype=float)", ...}
  }
}
```
