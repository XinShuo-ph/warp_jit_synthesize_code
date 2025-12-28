# GPU Analysis

## Current CUDA Support
- ir_extractor.py has device param: Yes (but only "cpu" in SUPPORTED_DEVICES)
- Tested with device="cuda": No GPU available (CUDA driver not found)

## CPU vs GPU IR Differences
| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| Header | `cpu_module_header` (no CUDA pragmas) | `cuda_module_header` (includes `__CUDACC__`, `__brkpt()`) |
| Kernel template | `cpu_kernel_template_forward/backward` | `cuda_kernel_template_forward/backward` |
| Thread ID | `wp::tid(task_index, dim)` | Device-side `threadIdx`, `blockIdx` |
| Memory model | Host memory, sequential task index | Device memory, CUDA grid/block model |

## Changes Needed for GPU
1. Add `"cuda"` to `SUPPORTED_DEVICES` in ir_extractor.py
2. Set `output_arch` to a valid CUDA arch (e.g., 75, 80, 86) instead of None
3. Call `builder.codegen("cuda")` instead of `builder.codegen("cpu")`
4. Handle case when CUDA not available (check `wp.is_cuda_available()`)

## Minimal Code Change
```python
# In ir_extractor.py
SUPPORTED_DEVICES: Final[set[str]] = {"cpu", "cuda"}

def extract_ir(kernel: wp.Kernel, device: str = "cpu") -> str:
    if device == "cuda" and not wp.is_cuda_available():
        raise RuntimeError("CUDA not available")
    
    builder_options = dict(module.options)
    builder_options["output_arch"] = None if device == "cpu" else 75  # e.g. SM75
    
    builder = ModuleBuilder(module, builder_options, ...)
    source = builder.codegen(device)  # "cpu" or "cuda"
    return source
```

## New GPU-Specific Patterns to Add
- [ ] Thread synchronization (`__syncthreads`, `wp.tile_sync()`)
- [ ] Shared memory usage patterns
- [ ] Warp-level primitives (shuffle, vote)
- [ ] Grid-stride loop patterns
- [ ] Memory coalescing patterns
