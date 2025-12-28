# GPU Analysis

## Current CUDA Support
- `ir_extractor.py` has device param: **Yes** (`device="cpu"` or `device="cuda"`)
- Tested with `device="cuda"`: **Pass** (codegen works even without GPU hardware)

## Key Finding
CUDA code generation via `codegen_kernel(kernel, device="cuda", ...)` works without requiring actual GPU hardware. The Warp library can generate CUDA source code on a CPU-only system.

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| **Function declaration** | `void kernel_cpu_kernel_forward(...)` | `extern "C" __global__ void kernel_cuda_kernel_forward(...)` |
| **Thread indexing** | `builtin_tid1d()` (same function name) | Uses `blockDim.x`, `blockIdx.x`, `threadIdx.x` in outer loop |
| **Argument passing** | Via struct pointer `*_wp_args` | Direct parameters (unpacked into function signature) |
| **Iteration** | No explicit loop (caller handles) | Grid-stride loop: `for (_idx = blockDim.x * blockIdx.x + threadIdx.x; ...)` |
| **Shared memory** | Not present | `wp::tile_shared_storage_t tile_mem;` initialized |
| **Control flow** | Uses `return;` | Uses `continue;` in grid-stride loop |
| **Code length** | ~2678 chars | ~3303 chars (25% larger) |

## Structural Differences

### CPU Pattern
```cpp
void kernel_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_kernel_HASH *_wp_args)
{
    // Unpack from struct
    wp::array_t<wp::float32> var_x = _wp_args->x;
    
    // ... kernel logic ...
}
```

### CUDA Pattern
```cpp
extern "C" __global__ void kernel_cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_x)  // Direct parameters
{
    wp::tile_shared_storage_t tile_mem;

    for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
         _idx < dim.size;
         _idx += blockDim.x * gridDim.x)
    {
        wp::tile_shared_storage_t::init();
        // ... kernel logic (same as CPU) ...
    }
}
```

## Changes Needed for Full GPU Support in ir_extractor.py

1. **No changes required** - The existing `device` parameter already supports `"cuda"`:
   ```python
   source = get_kernel_ir(my_kernel, device="cuda")
   ```

2. **Testing recommendation**: Add CUDA-specific tests to `test_extractor.py`:
   ```python
   def test_cuda_codegen():
       ir = get_kernel_ir(some_kernel, device="cuda")
       assert "__global__" in ir
       assert "_cuda_kernel_forward" in ir
   ```

## New GPU-Specific Patterns to Observe

When training on GPU code, the model should learn:

- [x] `__global__` function decoration pattern
- [x] Grid-stride loop idiom for parallelization
- [x] Direct parameter passing (vs struct unpacking)
- [x] `wp::tile_shared_storage_t` shared memory initialization
- [ ] Block synchronization (`__syncthreads()`) - not in simple kernels, may appear in reduction kernels
- [ ] Warp-level primitives - not observed in basic tests

## Recommendations for Data Generation

1. **Dual generation**: For each kernel, generate both CPU and CUDA IR to maximize training data variety
2. **Device-agnostic patterns**: Core logic (wp::add, wp::load, etc.) is identical between CPU and CUDA
3. **Platform-specific patterns**: Thread indexing and memory patterns differ significantly
4. **Simpler CUDA**: CUDA code may be easier to learn as it uses direct parameters instead of struct indirection

## Test Script

A test script was created at `code/extraction/test_cuda_codegen.py` that demonstrates:
- Side-by-side CPU vs CUDA code generation
- Key structural differences
- Both forward and backward passes
