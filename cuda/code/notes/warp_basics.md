# Warp Kernel Compilation Flow

## Overview
Warp compiles Python kernel functions to native C++/CUDA code at runtime via JIT.

## Compilation Pipeline
1. **Decoration**: `@wp.kernel` decorator creates a `Kernel` object
2. **AST Parsing**: `Adjoint` class parses Python source, builds AST (`kernel.adj`)
3. **First Launch**: Triggers module build when kernel is first called
4. **Build Phase**: `ModuleBuilder.build_kernel()` processes kernel dependencies
5. **Codegen**: `ModuleBuilder.codegen(device)` generates C++ source
6. **Native Compile**: Source compiled to shared library (cached in `~/.cache/warp/`)

## Key Classes (warp/_src/)
- `context.Kernel`: Holds kernel func, adjoint, module reference
- `context.Module`: Container for kernels/functions in a Python module  
- `context.ModuleBuilder`: Orchestrates code generation
- `codegen.Adjoint`: Parses Python source, stores AST, generates IR

## Accessing Generated Code
```python
from warp._src.context import ModuleHasher, ModuleBuilder

hasher = ModuleHasher(kernel.module)
options = {"block_dim": 256, "enable_backward": True, "mode": "release"}
builder = ModuleBuilder(kernel.module, options, hasher)
cpp_source = builder.codegen("cpu")  # or "cuda"
```

## Key Attributes
- `kernel.adj.source`: Original Python source code
- `kernel.adj.tree`: AST of parsed kernel
- `kernel.module`: Parent module containing the kernel
- Generated code includes forward AND backward (adjoint) kernels

## Cache Location
Compiled modules cached at: `~/.cache/warp/{version}/`
