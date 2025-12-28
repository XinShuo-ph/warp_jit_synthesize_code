# CUDA Adaptation Plan

## Goal
Enable synthesis of Python-to-CUDA-IR pairs using `warp` without requiring a physical GPU for generation.

## Analysis
- **Availability**: `warp` allows `codegen("cuda")` even without a CUDA driver.
- **Extraction**: The existing regex `void {func_name}` works for CUDA kernels because they are defined as `extern "C" __global__ void {func_name}`.
- **Pipeline**: Needs to be updated to expose the `device` parameter.

## Implementation Steps

1. **IR Extractor**:
   - Verify `device` parameter propagation.
   - Ensure "cuda" output is distinct and correctly formatted.

2. **Pipeline**:
   - Add `--device` CLI argument to `pipeline.py`.
   - Pass `device` to `synthesize_batch` and `synthesize_pair`.
   - Update metadata to reflect the correct device.

3. **Validation**:
   - Since we can't *run* the kernels, we will rely on:
     - Successful `codegen("cuda")`.
     - Successful regex extraction of the kernel body.
     - Visual inspection of generated `.cu` code (checking for `__global__`, thread indexing, etc.).

4. **User Deliverables**:
   - `cuda_tests/`: Scripts that the user can run on their GPU machine to verify the generated code actually compiles and runs.
