# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **No** (hardcoded to CPU)
- Tested with device="cuda": **No GPU available** in this environment
- Warp runs in CPU-only mode

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| File extension | `.cpp` | `.cu` |
| Forward function name | `{kernel}_cpu_kernel_forward` | `{kernel}_cuda_kernel_forward` |
| Backward function name | `{kernel}_cpu_kernel_backward` | `{kernel}_cuda_kernel_backward` |
| Function signature | `void name(dim, task_index, args*)` | `__global__ void name(args)` |
| Thread indexing | `task_index` parameter | `blockDim * blockIdx + threadIdx` |
| Includes | `#include "builtin.h"` | CUDA-specific headers |
| Loop structure | Sequential task loop | Grid-stride loop |

## Code Structure Comparison

### CPU (current)
```cpp
void kernel_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel *_wp_args)
{
    // argument loading
    // primal vars
    // forward pass
    var_0 = builtin_tid1d();
    // ...
}
```

### GPU (CUDA)
```cuda
extern "C" __global__ void kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp_args_kernel args)
{
    wp::tile_shared_storage_t tile_mem;
    
    for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
         _idx < dim.count();
         _idx += gridDim.x * blockDim.x)
    {
        // kernel body with _idx as thread id
    }
}
```

## Changes Needed for GPU

### 1. Update `ir_extractor.py`

Add device parameter to control extraction mode:

```python
def extract_ir(self, kernel, device: str = "cpu", force_compile: bool = True) -> Dict:
    """
    Args:
        device: "cpu" or "cuda" - which IR to extract
    """
    ...
```

### 2. Modify file search

```python
def _find_ir_file(self, kernel, device: str) -> Optional[str]:
    extension = ".cpp" if device == "cpu" else ".cu"
    # Search cache for matching file
    ...
```

### 3. Update function extraction patterns

```python
def _extract_function(self, code: str, device: str, direction: str) -> Optional[str]:
    suffix = f"{device}_kernel_{direction}"  # e.g., "cuda_kernel_forward"
    # Extract function matching this suffix
    ...
```

### 4. Handle CUDA-specific patterns

- Parse `__global__` function declarations
- Handle grid-stride loop structure
- Extract shared memory declarations

## New GPU-Specific Patterns to Add

- [ ] Grid-stride loop idiom (`blockDim.x * blockIdx.x + threadIdx.x`)
- [ ] Shared memory allocation (`__shared__` or `tile_shared_storage_t`)
- [ ] Warp-level primitives (`__syncthreads`, warp shuffles)
- [ ] Device-to-host data transfers (`wp::array_t` device pointers)
- [ ] Tensor core / tile operations

## Implementation Priority

1. **Low effort**: Add `device` parameter, change file extension and function suffix pattern
2. **Medium effort**: Parse CUDA-specific loop structures  
3. **High effort**: Generate synthetic CUDA kernels with realistic GPU idioms

## Testing Requirements

To test GPU IR extraction:
1. Install NVIDIA CUDA drivers
2. Ensure `nvidia-smi` works
3. Run: `wp.init(); wp.is_cuda_available()` should return True
4. Compile kernel with `device="cuda"`: `wp.launch(kernel, dim=n, inputs=[...], device="cuda")`
5. Cache should contain `.cu` files alongside `.cpp` files

## References

- Warp codegen: `/home/ubuntu/.local/lib/python3.12/site-packages/warp/_src/codegen.py`
- CPU template: `cpu_kernel_template_forward`
- CUDA template: `cuda_kernel_template_forward`
