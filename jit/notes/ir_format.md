# IR format (M2)

- **Chosen representation**: Warp-generated **CPU C++ source** for the kernel’s owning module (string).
- **How it’s obtained**: internal `warp._src.context.ModuleBuilder(...).codegen("cpu")`.
- **Why this**:
  - Pure string output (easy to serialize for Python→IR pairs).
  - Contains mangled kernel symbols (can assert presence for pairing).
  - Deterministic across repeated extraction runs (for the same Warp version + kernel source).
- **Caveats**:
  - It’s module-level codegen (may include multiple kernels/functions).
  - It’s not LLVM IR; it’s the generated C++ compilation unit used for CPU builds.
  - Depends on Warp internal APIs (pin `warp-lang` version for dataset generation).

