# Warp basics (M1)

- A Warp kernel is a Python function decorated with `@wp.kernel` and typed with Warp types (`wp.array`, scalars, structs).
- Calling `wp.launch(kernel, dim=..., inputs=[...], device=...)` triggers compilation the first time a given kernel/module is used on a given device.
- Warp builds a **module** for the Python file / kernel set; the load message shows a content-hash-like id and whether it was `(compiled)` or `(cached)`.
- Compiled artifacts are cached under the **kernel cache** directory (printed on init), e.g. `~/.cache/warp/<version>`.
- CPU-only runs are supported (this environment has no CUDA driver; `wp.is_cuda_available()` is `False`).

Where to study next for IR extraction (M2):
- `warp/context.py`: kernel/module build, device-specific compilation, caching, load lifecycle.
- `warp/codegen.py`: Python AST lowering and IR/code generation plumbing (key entry points for capturing intermediate forms).
- `warp/types.py`: type system that drives codegen and IR typing.

