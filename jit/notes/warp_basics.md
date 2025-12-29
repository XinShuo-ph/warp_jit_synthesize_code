# Warp Kernel Compilation Flow

## Python → C++/CUDA Pipeline

1. **Decoration**: `@wp.kernel` registers the Python function
2. **AST Parse**: `codegen.Adjoint` parses the source via `ast.parse()`
3. **Code Generation**: `codegen_kernel()` generates C++/CUDA code
4. **Compilation**: Native compiler (gcc/nvcc) compiles to `.o`/`.cubin`
5. **Caching**: Results cached in `~/.cache/warp/<version>/`

## Key Components

- `codegen.Adjoint`: Stores parsed AST, generates forward/reverse code
- `codegen_kernel(kernel, device, options)`: Main entry for kernel codegen
- `codegen_func_forward/reverse()`: Generate forward/backward passes
- `kernel.get_mangled_name()`: Unique name for compiled function

## IR Location

Generated code stored in cache dir: `~/.cache/warp/<version>/wp_<module>_<hash>/`
- `.cpp` - CPU source code
- `.cu` - CUDA source code (if GPU)
- `.o/.cubin` - Compiled objects

## Programmatic IR Access

```python
from warp._src.codegen import codegen_kernel, codegen_module
options = {"enable_backward": True}
cpp_code = codegen_kernel(kernel, "cpu", options) + codegen_module(kernel, "cpu", options)
```

## Forward/Backward Code

Warp auto-generates backward (adjoint) code for autodiff:
- Forward: Computes primal values
- Backward: Computes gradients via adj_ prefixed functions

## Cache Key

Hash includes: source code, types, device, options. Same kernel+inputs = same hash.
