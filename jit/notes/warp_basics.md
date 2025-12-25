# Warp Basics: Compilation and IR

## Compilation Flow
1. **Definition**: Python function is decorated with `@wp.kernel`. This creates a `warp.context.Function` object.
2. **Analysis**: The `Function` object initializes an `Adjoint` object, which parses the Python AST.
3. **Build**: When a kernel is launched or built, `ModuleBuilder` calls `Adjoint.build()`.
4. **IR Generation**: `Adjoint` traverses the AST and generates C++ code snippets, stored in `Adjoint.blocks`. This is the intermediate representation.
5. **Code Assembly**: `codegen_kernel` assembles these snippets, adds function signatures and struct definitions, producing a full `.cpp` or `.cu` source file.
6. **Compilation**: The source is compiled by the host compiler (GCC/MSVC) or NVRTC (for CUDA) into a shared library or PTX.
7. **Loading**: The compiled module is loaded and the kernel entry point is retrieved.

## Intermediate Representation (IR)
- **Location**: `kernel.adj.blocks`.
- **Structure**: A list of `Block` objects. Each `Block` contains `body_forward` (list of strings).
- **Format**: C++ statements using Warp runtime builtins (e.g., `wp::load`, `wp::mul`).
- **Variables**: SSA-style variables (`var_0`, `var_1`) are generated during AST traversal.

## Extraction Strategy
To extract the IR:
1. Initialize `ModuleBuilder`.
2. Call `builder.build_kernel(kernel)`.
3. Read `kernel.adj.blocks[i].body_forward`.
