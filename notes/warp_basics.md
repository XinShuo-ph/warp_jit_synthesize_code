# Warp Compilation Flow and IR Location

## Compilation Flow
1. **Kernel Definition**: `@wp.kernel` creates Kernel object
2. **First Launch**: wp.launch() triggers Module.compile()
3. **Code Generation**: codegen_kernel() in `_src/codegen.py` (line 4290)
   - Parses Python AST
   - Creates intermediate Var objects (line 673)
   - Generates C++ with explicit variables
4. **Output**: Argument struct + forward/backward functions
5. **Cache**: Saved to `~/.cache/warp/<version>/wp_<module>_<hash>/`
   - `.cpp`: Generated C++ source (the IR)
   - `.o`: Compiled object file
   - `.meta`: Metadata

## Key Files
- `warp/_src/context.py`: Module and Kernel classes
- `warp/_src/codegen.py`: Code generation (4391 lines)
- `warp/_src/types.py`: Type system

## IR Structure
```cpp
wp::int32 var_0;           // Thread ID
wp::float32 var_3;         // Intermediate result
var_0 = builtin_tid1d();   // Python: tid = wp.tid()
var_3 = wp::mul(var_4, var_2);  // Python: a[tid] * 2.0
```

## M2 Extraction Strategy
1. Compile kernel to trigger IR generation
2. Locate cache file using module name
3. Parse `.cpp` file to extract kernel function
4. Pair with original Python source
