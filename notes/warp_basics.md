# Warp Kernel Compilation Flow

## Overview
Warp is a Python framework for high-performance GPU computing that compiles Python code to C++/CUDA at runtime using JIT (Just-In-Time) compilation.

## Key Components

### 1. Kernel Definition (`@wp.kernel` decorator)
- Python functions decorated with `@wp.kernel` become GPU/CPU kernels
- Source code is extracted via `inspect.getsource()`
- AST (Abstract Syntax Tree) is built using Python's `ast` module

### 2. Adjoint Class (`warp/_src/codegen.py`)
- Core transformation engine that converts Python to intermediate representation
- Parses Python function into AST
- Builds SSA (Static Single Assignment) form
- Generates both forward and backward (adjoint) passes for automatic differentiation

### 3. Code Generation Flow
1. **Parse**: Python source → AST (via `ast.parse()`)
2. **Transform**: AST → SSA intermediate form (via `Adjoint` class)
3. **Codegen**: SSA → C++ code (via `codegen_kernel()`)
4. **Compile**: C++ → binary (.o file) (via system compiler)
5. **Cache**: Store compiled binaries in `~/.cache/warp/{version}/`

### 4. Module System
- Each Python module containing kernels maps to a Warp Module object
- Modules are hashed based on kernel source code
- Format: `wp_{module_name}_{hash}/`
- Cache contains: `.cpp` (source), `.o` (binary), `.meta` (metadata)

### 5. Kernel Launch
- `wp.launch(kernel, dim, inputs)` triggers compilation if needed
- Module is loaded on specified device (cpu/cuda)
- Compiled code is cached for future runs
- On cache hit, loading is instant (~1-2ms)

## Intermediate Representation (IR)

The generated C++ code contains:
- **Argument structs**: Type-safe parameter passing
- **Forward function**: Implements kernel logic
- **Backward function**: Automatic differentiation (adjoint)
- **SSA variables**: `var_0`, `var_1`, etc. for intermediate values
- **Comments**: Source line mappings for debugging

## Key Files in Warp Source
- `warp/_src/context.py`: Kernel class, Module management
- `warp/_src/codegen.py`: Adjoint class, code generation
- `warp/_src/types.py`: Type system (vec3, arrays, etc.)

## IR Extraction Method
Generated C++ files are stored in kernel cache:
```
~/.cache/warp/{version}/wp_{module_name}_{hash}/
  - wp_{module_name}_{hash}.cpp  # The IR we want!
  - wp_{module_name}_{hash}.o    # Compiled binary
  - wp_{module_name}_{hash}.meta # Metadata
```
