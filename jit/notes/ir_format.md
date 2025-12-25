# Warp IR Format

The "IR" extracted from Warp is C++ (CPU) or CUDA C++ (GPU) source code.

## Structure

### Header
```cpp
extern "C" __global__ void kernel_name(
    wp::launch_bounds_t dim,
    arg_type arg_name,
    ...
)
```

### Body
1. **Shared Memory**: `wp::tile_shared_storage_t tile_mem;`
2. **Thread Loop**:
   ```cpp
   for (size_t _idx = ...; _idx < dim.size; _idx += ...) {
       wp::tile_shared_storage_t::init();
       ...
   }
   ```
3. **Variable Declarations**:
   - Variables are declared at the top of the loop scope.
   - Naming convention: `var_0`, `var_1`, etc. (SSA-like).
4. **Logic**:
   - Operations use `wp::` built-in functions (e.g., `wp::load`, `wp::add`, `wp::mul`).
   - Control flow matches Python (if, for, while).
   - Comments often map back to Python source lines (e.g., `// a[tid] = ... <L 56>`).

## Types
- Arrays: `wp::array_t<type>`
- Scalars: `wp::int32`, `wp::float32`
- Structs: `MyStruct_hash` (generated struct names)

## Differences (CPU vs CUDA)
- CPU kernels have `size_t task_index` argument and run single task per call in the loop (or loop inside `cpu_forward`).
- CUDA kernels use `__global__` and standard CUDA grid striding.
