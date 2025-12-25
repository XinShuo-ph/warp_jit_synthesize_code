# Warp JIT + where “IR” lives (practical)

- **Compilation trigger**: first `wp.launch(...)` / `wp.launch_tiled(...)` on a given device causes a module load+compile if no cached binary exists.

- **Module ownership**: kernels/functions are grouped by Python module name (`kernel.func.__module__`).
  - Resolver: `warp._src.context.get_module("<py_module_name>")` returns a `warp._src.context.Module`.

- **Build pipeline (CPU)**:
  - `Module.load()` → (cache miss) → `Module.compile()`
  - `Module.compile()` builds a `ModuleBuilder(...)` then runs `builder.codegen("cpu")` to emit **generated C++** for all unique kernels/functions.
  - The C++ is compiled via `warp._src.build.build_cpu()` → `runtime.llvm.wp_compile_cpp(...)` → produces a `.o`
  - The `.o` is loaded via `runtime.llvm.wp_load_obj(...)`

- **Build pipeline (CUDA)**:
  - `Module.compile()` runs `builder.codegen("cuda")` to emit **generated CUDA C++** (`.cu`)
  - Compiled by `warp._src.build.build_cuda(...)` to **PTX or CUBIN** (see `Module.get_compile_output_name()`).

- **Where the generated artifacts land (kernel cache)**:
  - Cache root: `warp.config.kernel_cache_dir` (set before `wp.init()`), or env `WARP_CACHE_PATH`.
  - Per-module directory name: `Module.get_module_identifier()` → `wp_<module_name>_<hash7>` (unless `"strip_hash"` is enabled).
  - Inside `<kernel_cache_dir>/<module_id>/` you typically get:
    - `<module_id>.cpp` (CPU codegen “IR”)
    - `<module_id>.o` (CPU object)
    - `<module_id>.cu` and `<module_id>.smXX.ptx|cubin` (CUDA paths)
    - `<module_id>.meta` (kernel metadata, e.g. shared-mem bytes)

- **Practical “IR” for dataset extraction (M2+)**: treat the cached generated source (`.cpp` / `.cu`) and/or PTX as the JIT intermediate representation paired with the original Python kernel.

