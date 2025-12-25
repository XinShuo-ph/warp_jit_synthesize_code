# Warp Basics

## Kernel Compilation Flow
1. Python function decorated with `@wp.kernel` → `Kernel` object created
2. Source code extracted via `inspect.getsourcelines()` → stored in `kernel.adj.source`
3. AST parsed → stored in `kernel.adj.tree`
4. On first launch: `ModuleHasher` hashes kernel → `ModuleBuilder` generates code
5. C++/CUDA source written to cache: `~/.cache/warp/{version}/wp_{module}_{hash}/`
6. Code compiled to `.o` (CPU) or `.ptx/.cubin` (CUDA)

## Key Objects
- `kernel.adj` (Adjoint): Contains source, AST, arg types, forward/backward codegen
- `kernel.module` (Module): Contains all kernels/functions, manages compilation
- `kernel.module.execs`: Dict of compiled ModuleExec objects per device

## IR Location
- Cache dir: `~/.cache/warp/{version}/`
- Each module: `wp_{module_name}_{hash_prefix}/`
- Files: `.cpp` (CPU source), `.cu` (CUDA source), `.o`/`.cubin` (compiled)

## Generated IR Structure
```cpp
// 1. Args struct: wp_args_{kernel_name}_{hash}
struct wp_args_* { wp::array_t<T> arg_name; ... };

// 2. Forward kernel: {name}_cpu_kernel_forward / _cuda_kernel_forward
void {name}_cpu_kernel_forward(wp::launch_bounds_t dim, size_t task_index, wp_args_* *args);

// 3. Backward kernel: {name}_cpu_kernel_backward (auto-differentiation)
void {name}_cpu_kernel_backward(wp::launch_bounds_t dim, ..., wp_args_* *adj_args);

// 4. Entry points: extern "C" {name}_cpu_forward / _cuda_forward
```

## Extracting IR
Use `code/extraction/ir_extractor.py`:
- `get_kernel_source(kernel)` → Python source
- `extract_ir(kernel, device)` → Generated C++/CUDA
- `extract_pair(kernel, device)` → Dict with both
