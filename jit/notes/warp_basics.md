# Warp Compilation Basics

## Kernel Compilation Flow

1. **Definition**: 
   - Decorating a function with `@wp.kernel` creates a `Kernel` object.
   - A `warp.codegen.Adjoint` instance is created for the kernel, which parses the Python source into an AST.

2. **Launch & Trigger**:
   - `wp.launch()` invokes `context.Module.load()`.
   - If not already loaded, `Module.compile()` is triggered.

3. **Code Generation**:
   - `Module.compile()` initializes a `ModuleBuilder`.
   - `ModuleBuilder` calls `kernel.adj.build()`, which uses `Adjoint.eval()` to traverse the AST.
   - Visitor methods (`emit_Assign`, `emit_BinOp`, etc.) translate AST nodes into C++/CUDA source code strings.
   - These strings are stored in `Block` objects within the `Adjoint` instance (`body_forward`, `body_reverse`).
   - `ModuleBuilder` aggregates these snippets into a complete `.cpp` or `.cu` source file.

4. **Compilation**:
   - `warp.build.build_cpu()` or `build_cuda()` invokes the system compiler (Clang/MSVC or NVCC).
   - The resulting shared library or PTX is loaded back into Python.

## IR Location

- **Primary IR**: The Python AST serves as the initial representation.
- **Intermediate**: The `Adjoint` object holds the "IR" in the form of lists of C++ statement strings within `Block` objects (`adj.blocks`).
- **Lower Level**: The generated C++/CUDA source code is written to the build directory (e.g., `~/.cache/warp/...`) before being compiled to machine code/PTX.
- There is no exposed high-level IR (like MLIR) in the standard flow; the translation is effectively Python AST $\to$ C++ Source.
