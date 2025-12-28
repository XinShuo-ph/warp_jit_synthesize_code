# CUDA Backend Verification Instructions

This document provides instructions for verifying the CUDA backend adaptation on a GPU-enabled machine.

## Prerequisites

- NVIDIA GPU with CUDA drivers installed.
- Python 3.8+
- `warp-lang` installed (`pip install warp-lang`)

## 1. Verify Environment

Run the basic kernel tests to ensure Warp and CUDA are working correctly.

```bash
cd cuda/code
python3 examples/test_basic_kernels.py --device cuda
```

**Expected Output:**
- "Running Warp Basic Kernel Tests on cuda"
- 3/3 tests passed.

## 2. Verify Synthesis Pipeline (End-to-End)

Run the synthesis pipeline to generate Python->IR pairs using the CUDA backend.

```bash
cd cuda/code/synthesis
python3 pipeline.py --count 5 --device cuda --output ../../data/validation_cuda
```

**Expected Output:**
- "Starting synthesis pipeline..."
- "Device: cuda"
- "Successful: 5"
- JSON files generated in `cuda/data/validation_cuda/`

## 3. Validate Generated Data

Check that the generated JSON files contain valid C++ CUDA code.

```bash
# Inspect one of the generated files
cat ../../data/validation_cuda/*_arithmetic_*.json
```

**Checklist:**
- `cpp_ir_forward` should contain `__global__ void ...` or similar CUDA kernels (or device functions).
- `cpp_ir_full` should contain the full `.cu` source.

## Troubleshooting

- **CUDA driver not found**: Ensure `nvidia-smi` works and CUDA toolkit is in PATH.
- **IR extraction failed**: If the regex patterns don't match the generated CUDA code, check `cuda/code/extraction/ir_extractor.py` and adjust the regex in `extract_kernel_functions` (lines 70+).
