# IR format (M2)

- Extracted artifact: **Warp-generated JIT source** returned by `warp._src.context.ModuleBuilder.codegen(device)`.
- For CPU (`device="cpu"`), this is a single C++ translation unit string containing:
  - module header (`cpu_module_header`)
  - generated struct definitions
  - generated forward/reverse functions (if applicable)
  - generated kernel entrypoints + module registration glue
- For CUDA (`device="cuda"`), the same pipeline emits CUDA C++ source (and is later compiled to PTX/CUBIN by Warp).
- Determinism expectation: for a fixed Warp version + kernel source + module options, `codegen()` output should be byte-identical.

Primary extraction hook:
- `warp/_src/context.py`: `Module.compile()` constructs a `ModuleBuilder` and calls `builder.codegen("cpu"|"cuda")`.

