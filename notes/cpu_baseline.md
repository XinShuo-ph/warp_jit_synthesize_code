# CPU Baseline Analysis

## Overview
The current codebase is well-structured for CPU generation. 
The core components `ir_extractor.py` and `pipeline.py` already support a `device` parameter.

## Findings
1. **IR Extraction**: `ir_extractor.py`'s `extract_ir` function accepts a `device` argument.
2. **Pipeline**: `pipeline.py` has a `device` variable but it is defaulted to "cpu" and does not seem to be exposed via CLI.
3. **Batch Generator**: `batch_generator.py` hardcodes `"device": "cpu"`.
4. **Warp Capabilities**: Warp can generate CUDA IR (`.cu` code) even without a GPU present, as long as we don't try to *execute* the kernel.

## Plan for CUDA Adaptation
1. **Expose Device Parameter**: Add `--device` argument to `pipeline.py` and `batch_generator.py`.
2. **Update IR Extractor**: Ensure `ir_extractor.py` handles any CUDA-specific differences in function naming (e.g. `_cuda_kernel_forward` vs `_cpu_kernel_forward`).
   - *Check*: The code already uses f"{mangled_name}_{device}_kernel_forward", so it should work automatically.
3. **Verification**: Generate a few samples with `--device cuda` and inspect the output to confirm it looks like CUDA code (e.g., check for `__global__` or similar CUDA keywords if visible in the extracted IR).
