# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **No**
- Tested with device="cuda": **No GPU available in environment**
- Current implementation: CPU-only IR extraction

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| Function signature | `void {name}_cpu_kernel_forward(wp::launch_bounds_t dim, size_t task_index, args*)` | `extern "C" __global__ void {name}_cuda_kernel_forward(args)` |
| Thread ID | Uses `task_index` passed from host loop | Uses CUDA built-in `threadIdx`, `blockIdx` |
| Memory model | Host memory, sequential access | Device memory, parallel access |
| File extension | `.cpp` | `.cu` |
| Launch model | CPU loop over `dim.size` tasks | GPU grid/block launch |
| Header | `#define WP_TILE_BLOCK_DIM 1` | CUDA includes + `tile_shared_storage_t tile_mem;` |
| Compilation | Standard C++ compiler | NVCC/CUDA toolkit |

## Generated Code Structure

### CPU Kernel Template
```cpp
void {name}_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_{name} *_wp_args)
{
    // Variable declarations
    // Forward pass code
}

// Host launcher loops over all tasks
for (size_t task_index = 0; task_index < dim.size; ++task_index)
{
    {name}_cpu_kernel_forward(dim, task_index, _wp_args);
}
```

### CUDA Kernel Template
```cpp
extern "C" __global__ void {name}_cuda_kernel_forward(args)
{
    wp::tile_shared_storage_t tile_mem;
    // Shared memory setup
    // Parallel execution across GPU threads
}
```

## Changes Needed for GPU Support

1. **Add device parameter to ir_extractor**:
   - `extract_ir(kernel, device='cpu')` → `extract_ir(kernel, device='cuda')`
   - Use `wp.launch(..., device='cuda')` to trigger CUDA compilation

2. **Handle CUDA file caching**:
   - Look for `.cu` files instead of `.cpp` in cache
   - Cache structure might differ for PTX/CUBIN outputs

3. **Update cache file discovery**:
   - CUDA modules have different output formats (PTX for JIT, CUBIN for precompiled)
   - Check `wp.config.cuda_output` setting

4. **Environment requirements**:
   - CUDA toolkit must be installed
   - GPU device must be available (`wp.is_cuda_available()`)
   - NVIDIA driver compatibility

## New GPU-Specific Patterns to Add

- [ ] Shared memory usage (`__shared__` declarations)
- [ ] Atomic operations (`wp.atomic_add`, etc.)
- [ ] Warp-level primitives (shuffle operations)
- [ ] Thread synchronization (`wp.synchronize_device()`)
- [ ] Grid/block dimension variations
- [ ] Tile-based operations using `wp.tile_*` functions
- [ ] Multi-GPU patterns with explicit device specification

## Implementation Plan

### Phase 1: Basic CUDA IR Extraction
```python
def extract_ir(kernel, device='cpu'):
    if device == 'cuda' and not wp.is_cuda_available():
        raise RuntimeError("CUDA device not available")
    
    # Launch on specified device
    wp.launch(kernel, dim=1, inputs=dummy_inputs, device=device)
    
    # Find appropriate cache file (.cpp or .cu)
    extension = '.cu' if device == 'cuda' else '.cpp'
    cache_file = find_cache_file(module_name, extension)
```

### Phase 2: Unified Dataset Format
```json
{
  "python_source": "...",
  "ir_code": "...",
  "device": "cpu|cuda",
  "ir_type": "cpp|cu|ptx"
}
```

### Phase 3: GPU-Specific Kernels
Add generator patterns using:
- `wp.tile_load`, `wp.tile_store` for tile operations
- `wp.atomic_*` for atomic operations
- Multi-dimensional grids for 2D/3D problems
