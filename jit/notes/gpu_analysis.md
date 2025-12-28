# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: **Yes** (`device="cpu"` or `device="cuda"`)
- Tested with device="cuda": **Pass** (codegen works, no GPU needed for extraction)

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| Function signature | `static void kernel_forward(...)` | `static CUDA_CALLABLE void kernel_forward(...)` |
| Body code | Identical | Identical |
| Memory ops | `wp::address`, `wp::load`, `wp::array_store` | Same primitives |
| Thread ID | `builtin_tid1d()` | Same function (resolved at compile) |
| Control flow | Standard C++ | Same |

### Key Finding
The IR extracted is **nearly identical** between CPU and CUDA targets. The only difference is the `CUDA_CALLABLE` macro in the function signature, which expands to `__device__` for NVCC compilation.

This is because Warp's code generation produces **portable C++ code** that can be compiled for either target. The `wp::` namespace functions are implemented differently per platform but the kernel source is shared.

## Changes Needed for GPU Dataset Generation

1. **Already Supported**: Change `device="cpu"` to `device="cuda"` in pipeline calls
   ```python
   ir_map = get_kernel_ir(kernel, device="cuda")
   ```

2. **Dataset Schema Update**: Add or modify the `device` field in samples
   ```json
   {
     "device": "cuda",
     "cpp_source_forward": "static CUDA_CALLABLE void..."
   }
   ```

3. **Dual-Target Generation**: Generate both CPU and CUDA variants per kernel
   ```python
   sample = {
       "cpu_forward": get_kernel_ir(kernel, device="cpu")["forward"],
       "cuda_forward": get_kernel_ir(kernel, device="cuda")["forward"],
   }
   ```

## New GPU-Specific Patterns to Add

For richer CUDA-specific dataset, consider adding:

- [ ] **Shared memory** (`wp.tile_*` operations) - requires Warp 1.0+
- [ ] **Warp shuffles** - inter-thread communication
- [ ] **Atomic operations** - `wp.atomic_add`, `wp.atomic_sub`
- [ ] **Grid-stride loops** - processing arrays larger than grid size
- [ ] **Multi-dimensional kernels** - `wp.tid()` returning vec2/vec3

## Verification (No GPU Required)

The IR extraction for CUDA works without a physical GPU because:
1. Warp's `codegen_func()` only generates **source code strings**
2. No actual NVCC compilation occurs during extraction
3. The `device` parameter only affects codegen output format

```python
# This works even on CPU-only machines:
ir_cuda = get_kernel_ir(kernel, device="cuda")
# Returns valid CUDA-compatible C++ source
```

## Recommendations

1. **Quick Win**: Run `pipeline.py` with `device="cuda"` to generate CUDA dataset
2. **Dual Dataset**: Modify pipeline to output both CPU and CUDA for each kernel
3. **Future**: Add GPU-specific operations when targeting CUDA compilation
