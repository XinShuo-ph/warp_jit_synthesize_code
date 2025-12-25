# Warp Kernel Compilation Flow

## Compilation Pipeline
1. `@wp.kernel` decorator creates `Kernel` object with `Adjoint` for AST analysis
2. Python source → AST → SSA-form C++/CUDA code
3. Code cached at `~/.cache/warp/{version}/wp_{module}_{hash}/`
4. C++ compiled to `.o` (CPU) or PTX/cubin (CUDA)

## Key Structures
- `Kernel.adj.source` - Original Python source
- `Kernel.module` - Parent module managing compilation
- Generated files: `{hash}.cpp` (forward+backward), `{hash}.meta`, `{hash}.o`

## Generated IR Format
```cpp
// Forward kernel - executes primal computation
void kernel_cpu_kernel_forward(wp::launch_bounds_t dim, size_t task_index, args);

// Backward kernel - computes adjoints for autodiff
void kernel_cpu_kernel_backward(wp::launch_bounds_t dim, size_t task_index, args, adj_args);
```

## IR Extraction
- Python source: `kernel.adj.source`
- C++ IR: Read from cache `{cache_dir}/{module_id}/{module_id}.cpp`
- Module ID: `kernel.module.get_module_identifier()`

## Key Files
- `warp/_src/codegen.py` - Adjoint class, AST→IR transform
- `warp/_src/context.py` - Kernel, Module, ModuleBuilder classes
- `warp/_src/build.py` - Compilation to native code
