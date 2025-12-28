# GPU Analysis

## Current CUDA Support

| Aspect | Status |
|--------|--------|
| ir_extractor.py has device param | **Yes** - `extract_ir(kernel, device="cuda")` works |
| Tested with device="cuda" | **Pass** - Code generation works without GPU |
| Runtime execution tested | No GPU available for execution testing |

## Key Finding

**CUDA code generation works without a GPU!** The `device="cuda"` parameter generates CUDA C++ code at codegen time. Only kernel execution requires a GPU.

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| **File Extension** | `.cpp` | `.cu` |
| **Function Decorator** | `void func(...)` | `extern "C" __global__ void func(...)` |
| **Thread Index** | `task_index` parameter | `blockDim.x * blockIdx.x + threadIdx.x` calculation |
| **Arguments** | Pointer to args struct `*_wp_args` | Direct parameters in function signature |
| **Parallelization** | External task scheduler | CUDA grid-stride loop |
| **Shared Memory** | Not used | `wp::tile_shared_storage_t tile_mem` |
| **Built-in tid()** | `wp::tid(task_index, dim)` | `wp::tid(_idx, dim)` |

### CPU Forward Function Pattern
```cpp
void kernel_add_..._cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,                    // Thread ID passed in
    wp_args_kernel_add_... *_wp_args)     // Args via struct pointer
{
    // Unpack arguments from struct
    wp::array_t<wp::float32> var_a = _wp_args->a;
    
    // Computation...
    var_0 = builtin_tid1d();  // Uses task_index
}
```

### CUDA Forward Function Pattern
```cpp
extern "C" __global__ void kernel_add_..._cuda_kernel_forward(
    wp::launch_bounds_t dim,
    wp::array_t<wp::float32> var_a,       // Args passed directly
    wp::array_t<wp::float32> var_b,
    wp::array_t<wp::float32> var_c)
{
    wp::tile_shared_storage_t tile_mem;   // Shared memory
    
    // Grid-stride loop
    for (size_t _idx = blockDim.x * blockIdx.x + threadIdx.x;
         _idx < dim.size;
         _idx += blockDim.x * gridDim.x)
    {
        wp::tile_shared_storage_t::init();
        
        // Computation...
        var_0 = builtin_tid1d();  // Uses _idx
    }
}
```

## Header Differences

| Aspect | CPU | CUDA |
|--------|-----|------|
| **Breakpoint macro** | Not defined | `#define __debugbreak() __brkpt()` |
| **tid macros** | Use `task_index` | Use `_idx` |

## Identical Patterns (No Changes Needed)

The following operations are **identical** between CPU and CUDA:
- `wp::address(array, index)` - Get element pointer
- `wp::load(ptr)` - Load value
- `wp::array_store(array, index, value)` - Store value
- `wp::add/sub/mul/div/...` - Arithmetic operations
- `wp::sin/cos/exp/sqrt/...` - Math functions
- `wp::atomic_add/max/...` - Atomic operations
- `wp::dot/cross/normalize/...` - Vector operations
- Line comments with Python source references

## Changes Needed for GPU Dataset

### Minimal Changes
1. **Pipeline device parameter**: Add `--device cuda` CLI option
2. **Metadata update**: Change `"device": "cpu"` to `"device": "cuda"` in JSON output
3. **Validation**: Currently no changes needed - codegen already works

### Code Changes Required

```python
# In pipeline.py, update synthesize_pair():
def synthesize_pair(spec: KernelSpec, device: str = "cpu") -> dict[str, Any]:
    # ... existing code works for both devices ...
    # Just pass device parameter through
```

### Example Command
```bash
# Generate CPU pairs (current)
python3 code/synthesis/pipeline.py -n 100 --device cpu

# Generate CUDA pairs (new option)
python3 code/synthesis/pipeline.py -n 100 --device cuda
```

## New GPU-Specific Patterns to Add

The current kernel categories don't include GPU-specific patterns. Future additions could include:

- [ ] **Tiled operations**: Using `wp.tile()` for shared memory optimization
- [ ] **Block sync**: Patterns using `wp.synchronize_block()`
- [ ] **Warp operations**: Using `wp.warp_*()` primitives
- [ ] **Cooperative groups**: Multi-block coordination patterns
- [ ] **Texture memory**: Using texture/surface references
- [ ] **Constant memory**: GPU constant memory patterns

## Summary

| Task | Status |
|------|--------|
| CUDA codegen works | ✅ Verified |
| ir_extractor supports device param | ✅ Yes |
| Pipeline supports CUDA generation | ✅ Works (just pass device="cuda") |
| Changes needed | 🔧 Minimal - just add CLI --device option |
| GPU execution testing | ⏳ Requires GPU hardware |

## Implementation Priority

1. **Low effort, high value**: Add `--device` CLI option to pipeline.py
2. **Medium effort**: Generate mixed CPU+CUDA dataset pairs
3. **Higher effort**: Add GPU-specific kernel patterns (tiling, shared memory, etc.)
