# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **Yes** (`device="cpu"` or `device="cuda"`)
- Tested with device="cuda": **Pass** (generates CUDA IR even without GPU)
- GPU hardware available: **No** (CPU-only environment)

## Key Finding
Warp's `codegen_kernel()` function generates CUDA source code regardless of whether a GPU is present. The code generation is purely source-to-source transformation and does not require CUDA runtime or GPU hardware. This means our synthesis pipeline already works for both CPU and CUDA IR extraction.

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| Function signature | `void kernel_cpu_kernel_forward(dim, task_index, *args)` | `extern "C" __global__ void kernel_cuda_kernel_forward(dim, args...)` |
| Arguments | Packed in struct (`wp_args_*`), passed via pointer | Passed directly as function parameters |
| Thread indexing | `task_index` parameter provided by caller | Grid-stride loop with `blockDim.x * blockIdx.x + threadIdx.x` |
| Parallelism | Single task per call (caller loops) | All threads in kernel, grid-stride loop internal |
| Shared memory | Not explicitly initialized | `wp::tile_shared_storage_t tile_mem;` declared and `init()` called |
| Function prefix | None | `extern "C" __global__` |
| File extension | `.cpp` | `.cu` |

## Detailed Comparison

### CPU IR Structure
```cpp
struct wp_args_kernel_hash {
    wp::array_t<type> arg1;
    // ... all args packed in struct
};

void kernel_hash_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,           // CPU gets task index from caller
    wp_args_kernel_hash *_wp_args)
{
    // Unpack arguments from struct
    wp::array_t<type> var_arg1 = _wp_args->arg1;
    
    // Variable declarations
    // ...
    
    // Forward pass (single task)
    // ...
}
```

### CUDA IR Structure
```cuda
extern "C" __global__ void kernel_hash_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<type> var_arg1,   // Args passed directly
    // ... all args as parameters
) {
    wp::tile_shared_storage_t tile_mem;

    // Grid-stride loop
    for (size_t _idx = static_cast<size_t>(blockDim.x) * static_cast<size_t>(blockIdx.x) + static_cast<size_t>(threadIdx.x);
         _idx < dim.size;
         _idx += static_cast<size_t>(blockDim.x) * static_cast<size_t>(gridDim.x))
    {
        wp::tile_shared_storage_t::init();  // Shared memory reset
        
        // Variable declarations
        // ...
        
        // Forward pass
        // ...
    }
}
```

## Changes Needed for GPU
1. **None for IR extraction** - Current `ir_extractor.py` already supports both CPU and CUDA via `device` parameter
2. **None for pipeline** - Pipeline already tries CUDA first, falls back to CPU
3. **For training data diversity** - Could generate both CPU and CUDA IR for each kernel to double dataset size

## Recommendations for GPU-Specific Data

### Option 1: Dual Output (Recommended)
Modify `pipeline.py` to save both CPU and CUDA IR for each kernel:
```python
result = {
    "id": idx,
    "kernel_name": name,
    "python_code": code,
    "ir_cpu": get_kernel_ir(kernel, device="cpu"),
    "ir_cuda": get_kernel_ir(kernel, device="cuda")
}
```

### Option 2: Device as Input Feature
Add device as a conditioning input for the model:
```python
# Two samples per kernel
{"python_code": code, "device": "cpu", "ir_code": cpu_ir}
{"python_code": code, "device": "cuda", "ir_code": cuda_ir}
```

## New GPU-Specific Patterns to Add
- [ ] `wp.tile()` operations for tiled/block-level parallelism
- [ ] `wp.tile_map()` for tile-level operations
- [ ] Shared memory operations (`wp.tile_shared()`)
- [ ] Warp-level primitives if supported
- [ ] Multi-dimensional thread indexing (`wp.tid()` with multiple dims)

## Warp Source References
- `warp/_src/codegen.py` - Main code generation logic
- `warp/_src/codegen.py:codegen_kernel()` - Entry point for kernel code generation
- `warp/_src/context.py:Kernel` - Kernel object that stores generated code
- CPU template uses struct packing for arguments
- CUDA template uses direct parameter passing with grid-stride loop

## Conclusion
The current implementation is **GPU-ready** for IR extraction. No code changes are required to generate CUDA code. The main enhancement opportunity is generating both CPU and CUDA IR to create a richer training dataset.
