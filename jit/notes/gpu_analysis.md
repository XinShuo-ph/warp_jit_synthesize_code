# GPU Analysis

## Current CUDA Support
- `ir_extractor.py` has device param: **Yes** (`device="cpu"` or `device="cuda"`)
- Tested with `device="cuda"`: **No GPU available** (environment lacks CUDA driver)

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| Header | `cpu_module_header` | `cuda_module_header` with `__CUDACC__` debug support |
| Thread ID macro | `wp::tid(task_index, dim)` | `wp::tid(_idx, dim)` where `_idx` is CUDA thread index |
| Function template | `static {return_type} {name}(...)` | `static CUDA_CALLABLE {return_type} {name}(...)` |
| Kernel template | `void {name}_cpu_kernel_forward(...)` | `__global__ void {name}_cuda_kernel_forward(...)` |
| Thread loop | Sequential via `task_index` | CUDA grid-stride: `for (_idx = blockDim.x * blockIdx.x + threadIdx.x; ...)` |
| Module registration | `extern "C"` C-linkage wrappers | Same, but CUDA runtime handles kernel launch |
| Shared memory | `wp::tile_shared_storage_t` (stack-based) | `wp::tile_shared_storage_t` (CUDA shared memory) |

## Key Code Generation Paths in Warp

1. **`codegen.py`**: Contains both `cpu_*` and `cuda_*` templates
   - `cpu_module_header` / `cuda_module_header` (lines 3568-3613)
   - `cpu_kernel_template_forward` / `cuda_kernel_template_forward` (lines 3685-3731)
   - `codegen_func_forward()` handles device-specific indentation

2. **`context.py`**: `ModuleBuilder.codegen(device)` dispatches based on device
   - Adds appropriate header (cpu/cuda) at line 2031-2035
   - Returns raw source string for compilation

## Changes Needed for GPU Synthesis

1. **Environment**: Need CUDA driver + GPU hardware to test `device="cuda"` path
2. **Validation**: Add CUDA-specific test cases when GPU available
3. **No code changes needed**: The `extract_ir()` function already supports CUDA via the `device` parameter

## New GPU-Specific Patterns to Add (Future Work)

- [ ] Shared memory tiling patterns (`wp.tile_*` operations)
- [ ] Warp-level primitives (warp shuffle, ballot)
- [ ] Memory coalescing patterns
- [ ] Occupancy-aware kernel sizing
- [ ] Multi-GPU launch patterns (`device="cuda:N"`)

## Testing Recommendations

When GPU becomes available:
```bash
# Test CUDA extraction
python3 -c "
from extraction.ir_extractor import extract_ir
from extraction.cases.case_arith import get_kernel
ir = extract_ir(get_kernel(), device='cuda')
print(ir[:500])
"
```

The CUDA IR should contain `__global__` kernel declarations and grid-stride loops.
