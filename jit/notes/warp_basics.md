# Warp basics (JIT compilation + artifacts)

- `@wp.kernel` / `@wp.func` build a typed IR from Python source (Warp’s codegen).
- First `wp.launch(...)` triggers JIT compilation of the owning module on the chosen device.
  - **CPU**: generates native code via Warp’s LLVM toolchain.
  - **CUDA** (if available): generates device code (PTX/cubin) via NVRTC/toolchain.
- Compiled artifacts are cached between runs:
  - Cache root is printed at init (e.g. `~/.cache/warp/<version>`).
  - Override cache root with `WARP_CACHE_PATH` (see `warp._src.config.kernel_cache_dir` docs).
- Useful knobs when exploring compilation:
  - `wp.config.verbose = True` for detailed codegen/compile logs.
  - `wp.set_module_options({...})` to control per-module compilation (e.g. `enable_backward`, `lineinfo`, debug mode).
- Early “IR” to target in M2:
  - Generated source/compile units and cached module artifacts under the kernel cache directory (subdirs prefixed `wp_`).
  - Warp’s internal typed graph / generated code produced by `warp.codegen` during module build.

