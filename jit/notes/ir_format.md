# IR format (what we extract)

- **Default IR (CPU)**: Warp-generated C++ source from the kernel cache: `<module_id>.cpp`.
- **Optional IR (CUDA)**: Warp-generated CUDA source `<module_id>.cu` and/or compiled PTX `<module_id>.sm<arch>.ptx` when CUDA is available.

- **Where it comes from**: Warp codegen (`warp._src.context.Module.compile()` → `ModuleBuilder.codegen("cpu"/"cuda")`) writes these files into `warp.config.kernel_cache_dir`.
- **Module directory name**: `Module.get_module_identifier()` → `wp_<python_module_name>_<hash7>` (unless `"strip_hash"` is enabled).
- **Metadata file**: `<module_id>.meta` (JSON) is emitted alongside code, e.g., shared-memory bytes.

