# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **Yes** (default="cpu", accepts "cuda")
- Tested with device="cuda": **No GPU available** (CPU-only environment)

## ir_extractor.py CUDA Handling

The `extract_ir()` function already attempts CUDA codegen:

```python
# Generate CPU code
cpp_kernel = codegen_kernel(kernel, "cpu", options)
cpp_module = codegen_module(kernel, "cpu", options)
cpp_code = cpp_kernel + "\n" + cpp_module

# Try to generate CUDA code (may fail on CPU-only systems)
cuda_code = None
try:
    cuda_kernel = codegen_kernel(kernel, "cuda", options)
    cuda_code = cuda_kernel
except Exception:
    pass  # CUDA codegen may not be available
```

## CPU vs GPU IR Differences

Based on Warp source code analysis (`warp/_src/codegen.py`):

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| Thread indexing | `builtin_tid1d()` | `blockIdx * blockDim + threadIdx` |
| Entry point | `void kernel_cpu_forward(...)` | `__global__ void kernel_cuda_forward(...)` |
| Memory qualifiers | None | `__device__`, `__shared__` |
| Launch mechanism | Sequential for-loop | CUDA grid/block launch |
| Atomics | `wp::atomic_add` (CPU impl) | `atomicAdd` (CUDA impl) |
| Shared memory | Stack-allocated tiles | `__shared__` memory |
| Function decoration | None | `__device__` for device functions |

## Code Structure Comparison

### CPU Entry Point
```cpp
extern "C" {
WP_API void kernel_cpu_forward(wp::launch_bounds_t dim, wp_args *args) {
    for (size_t task_index = 0; task_index < dim.size; ++task_index) {
        kernel_cpu_kernel_forward(dim, task_index, args);
    }
}
}
```

### GPU Entry Point (expected)
```cuda
__global__ void kernel_cuda_forward(wp::launch_bounds_t dim, wp_args *args) {
    size_t task_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (task_index < dim.size) {
        // kernel body inline
    }
}
```

## Changes Needed for GPU Training Data

1. **Enable CUDA codegen**: Install CUDA toolkit, run on GPU-enabled system
2. **Store both outputs**: Modify data format to include `cuda_code` field alongside `cpp_code`
3. **Update ExtractedIR**: Already has `cuda_code` field (Optional[str])
4. **Extend generators**: Current kernels should work on GPU without changes
5. **Handle device-specific patterns**: Some Warp ops have different GPU implementations

## New GPU-Specific Patterns to Add

- [ ] Tile operations (`wp.tile_load`, `wp.tile_store`, `wp.tile_matmul`)
- [ ] Shared memory usage patterns
- [ ] Warp-level primitives (`wp.simt_ballot`, etc.)
- [ ] Multi-dimensional thread indexing (`wp.tid()` returns tuple on 2D/3D grids)
- [ ] Texture memory operations
- [ ] Cooperative groups patterns

## Recommended Next Steps

1. Run on GPU-enabled machine with `device="cuda"`
2. Verify CUDA codegen produces valid output
3. Update `generate_single_pair()` to capture both CPU and CUDA code:
   ```python
   result = {
       "id": pair_id,
       "kernel_name": kernel_name,
       "python": kernel_source.strip(),
       "cpp": ir.cpp_code,
       "cuda": ir.cuda_code,  # Add this
       "type": generator.__name__
   }
   ```
4. Create GPU-specific kernel generators for tile/shared memory patterns
