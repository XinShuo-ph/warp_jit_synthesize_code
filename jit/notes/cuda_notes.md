# CUDA Backend Notes

## Overview

The CUDA backend generates `.cu` style kernel code that runs on NVIDIA GPUs. The extraction pipeline supports both CPU (`.cpp`) and CUDA (`.cu`) code generation using the same kernel definitions.

## CPU vs CUDA Differences

| Aspect | CPU (.cpp) | CUDA (.cu) |
|--------|------------|------------|
| Function suffix | `_cpu_kernel_forward` | `_cuda_kernel_forward` |
| Thread ID | `builtin_tid1d()` | `blockDim.x * blockIdx.x + threadIdx.x` |
| Loop structure | `for (task_index...)` | Grid-stride loop |
| Argument passing | Via struct pointer | Direct parameters |
| Memory access | Standard pointers | Device memory pointers |
| Shared memory | Not used | `tile_shared_storage_t` |

## CUDA-Specific Code Structure

### Forward Kernel Signature
```cpp
void kernel_name_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<float> var_a,
    wp::array_t<float> var_b,
    wp::array_t<float> var_c)
```

### CUDA Threading Model
```cpp
for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) 
                   + static_cast<size_t>(threadIdx.x);
     _idx < dim.size;
     _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
```

### Backward Kernel Structure
```cpp
void kernel_name_cuda_kernel_backward(
    wp::launch_bounds_t dim,
    wp::array_t<float> var_a,    // Primal inputs
    wp::array_t<float> adj_a)    // Adjoint inputs
{
    // Forward pass (recompute)
    // ...
    // Reverse pass
    wp::adj_array_store(...);
}
```

## Supported Kernel Types

| Type | Forward | Backward | Notes |
|------|---------|----------|-------|
| arithmetic | ✓ | ✓ | Basic +, -, *, / |
| vector | ✓ | ✓ | wp.vec2/3/4 operations |
| matrix | ✓ | ✓ | wp.mat operations |
| control_flow | ✓ | ✓ | if/for loops |
| math | ✓ | ✓ | sin, cos, exp, log |
| atomic | ✓ | ✓ | atomic_add/min/max |
| nested_loop | ✓ | ✓ | Nested for loops |
| multi_condition | ✓ | ✓ | if/elif/else |
| combined | ✓ | ✓ | Mixed patterns |
| scalar_param | ✓ | ✓ | Scalar parameters |

## Pipeline Usage

### Generate CUDA IR
```bash
python3 jit/code/synthesis/pipeline.py -n 100 -d cuda -o data/cuda_samples
```

### Generate CUDA IR with backward pass
```bash
python3 jit/code/synthesis/pipeline.py -n 100 -d cuda -b -o data/cuda_samples
```

### Output Format
```json
{
  "python_source": "@wp.kernel\ndef kernel_name(...): ...",
  "ir_forward": "void kernel_name_cuda_kernel_forward(...) { ... }",
  "ir_backward": "void kernel_name_cuda_kernel_backward(...) { ... }",
  "metadata": {
    "device": "cuda",
    "ir_type": "cuda",
    "has_backward": true
  }
}
```

## Testing

### Without GPU (extraction only)
```bash
python3 -m pytest jit/tests/cuda/test_extraction.py -v
```

### With GPU (full validation)
```bash
./jit/tests/cuda/run_gpu_tests.sh
```

## Notes

1. **No GPU required for IR extraction**: Warp can generate CUDA code even in CPU-only mode
2. **Warp handles device abstraction**: Same Python kernel works for both CPU and CUDA
3. **Backward pass is optional**: Use `-b` flag to include adjoint kernels
4. **Thread model**: CUDA uses grid-stride loops for work distribution
