# GPU Analysis

## Current CUDA Support
- `ir_extractor.py` has device param: **Yes** (line 23: `device: str = "cpu"`)
- Tested with device="cuda": **No GPU available** (environment has no CUDA driver)
- Warp reports: "CUDA driver not found or failed to initialize"

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| **File extension** | `.cpp` | `.cu` |
| **Function decorator** | None | `extern "C" __global__` |
| **Thread indexing** | Sequential loop: `for (task_index = 0; task_index < dim.size; ++task_index)` | Grid/block parallel: `blockDim.x * blockIdx.x + threadIdx.x` |
| **Entry point pattern** | `{name}_cpu_kernel_forward(dim, task_index, args)` | `{name}_cuda_kernel_forward(dim, args)` (no task_index - uses threadIdx) |
| **Parallelization** | External (OpenMP or single thread) | Built-in (CUDA blocks/threads) |
| **Shared memory** | `wp::tile_shared_storage_t` in stack | `__shared__` memory |
| **Iteration style** | Host-side loop over task_index | Device-side grid-stride loop |
| **Module header** | `#define WP_NO_CRT` | Same + CUDA debug/breakpoint macros |

## Key Code Template Differences

### CPU Forward Kernel
```cpp
void {name}_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_{name} *_wp_args)
{
    // task_index passed as parameter
    // forward body here
}
```

### CUDA Forward Kernel  
```cpp
extern "C" __global__ void {name}_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp_args_{name} *_wp_args)
{
    wp::tile_shared_storage_t tile_mem;
    
    for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
         _idx < dim.size;
         _idx += blockDim.x * gridDim.x)
    {
        wp::tile_shared_storage_t::init();
        // forward body here (uses _idx instead of task_index)
    }
}
```

## Changes Needed for GPU Data Collection

1. **Environment**: Need GPU with CUDA driver installed
2. **Extraction code**: Already supports `device="cuda"` parameter
3. **Cache path**: Same structure, but generates `.cu` instead of `.cpp`
4. **Regex patterns**: Update pattern matching from `_cpu_kernel_` to `_cuda_kernel_`:
   ```python
   # Current CPU pattern
   rf'void\s+{kernel_name}_[a-f0-9]+_cpu_kernel_forward\('
   
   # New CUDA pattern
   rf'extern "C" __global__ void\s+{kernel_name}_[a-f0-9]+_cuda_kernel_forward\('
   ```

## New GPU-Specific Patterns to Add

- [ ] Grid-stride loop pattern extraction
- [ ] Shared memory allocation patterns (`__shared__`)  
- [ ] Synchronization primitives (`__syncthreads()`)
- [ ] Warp-level primitives (`__shfl_*`, etc.)
- [ ] Atomic operations (different CUDA atomics)
- [ ] Memory coalescing patterns

## Implementation Recommendations

1. **Dual extraction**: Modify pipeline to extract both CPU and CUDA IR when GPU available:
   ```python
   def extract_ir(kernel, device="cpu"):
       # existing code
   
   def extract_both(kernel):
       cpu_ir = extract_ir(kernel, "cpu")
       try:
           cuda_ir = extract_ir(kernel, "cuda")
       except:
           cuda_ir = None
       return cpu_ir, cuda_ir
   ```

2. **Data format extension**: Add `cuda_ir_forward` and `cuda_ir_backward` fields to JSON

3. **Kernel type additions** for GPU-specific patterns:
   - `reduction` - Parallel reduction with shared memory
   - `scan` - Prefix sum patterns
   - `stencil` - Neighbor access patterns with halo regions
   - `matrix_tile` - Tiled matrix operations

## Testing Without GPU

For development/testing without a GPU:
- Use `device="cpu"` (current approach)
- The generated CPU IR has nearly identical computation logic
- Main difference is the parallelization wrapper code
- Core kernel body (`forward_body`, `reverse_body`) is device-agnostic
