# CUDA Production Notes

## Offline Generation Capability
We have confirmed that `warp-lang` allows generating CUDA C++ source code without an active GPU or CUDA driver.

### Process
1. **Init**: `warp.init()` is called. It detects no GPU and warns, but proceeds in CPU mode.
2. **Compilation**: We create a `ModuleBuilder` and call `codegen(device="cuda")`.
3. **Result**: This produces C++ code containing CUDA intrinsics (`blockDim`, `threadIdx`, `atomicAdd`, etc.).

### Limitations
- **No PTX**: Since `nvcc` is not available in this environment, we cannot compile the C++ to PTX or CUBIN. The pipeline stops at the source code level.
- **No Execution**: We cannot run the kernels to verify numerical correctness.
- **Validation**: We rely on regex matching of CUDA keywords to verify that the output is indeed intended for CUDA.

### Dataset `cuda_v1`
- **Count**: 100 pairs
- **Location**: `data/cuda_v1/`
- **Format**:
  ```json
  {
    "python_source": "...",
    "cpp_forward": "... void ..._cuda_kernel_forward(...) ...",
    "metadata": { "device": "cuda", ... }
  }
  ```
