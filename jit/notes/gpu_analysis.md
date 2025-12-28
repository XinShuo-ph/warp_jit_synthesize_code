# GPU Analysis

## Current CUDA Support
- **ir_extractor.py has device param**: Yes (`device: str | None = "cpu"`)
- **Tested with device="cuda"**: No GPU available - Cannot test
- **Prefer parameter supports CUDA**: Yes (`prefer=("cpp", "cu", "ptx")`)
- **PTX/CUBIN arch handling**: Implemented via `output_arch` parameter

## CPU vs GPU IR Differences

| Aspect | CPU (.cpp) | GPU (.cu) |
|--------|------------|-----------|
| **File extension** | `.cpp` | `.cu` |
| **Kernel decorator** | `void func_cpu_kernel_forward(...)` | `extern "C" __global__ void func_cuda_kernel_forward(...)` |
| **Thread indexing** | `task_index` passed as argument | `blockDim.x * blockIdx.x + threadIdx.x` computed inline |
| **Args passing** | Via `wp_args_*` struct pointer | Direct function arguments |
| **Loop structure** | Single invocation per task | Grid-stride loop over `_idx` |
| **Shared memory** | `wp::tile_shared_storage_t tile_mem;` | Same, but with block-level scope |
| **Backward pass args** | `_wp_args, _wp_adj_args` pointers | All args + adj_args as direct params |
| **Entry point** | `extern "C" WP_API void {name}_cpu_forward(...)` | Kernel launch via CUDA runtime |
| **Compiled output** | `.o` (object file) | `.ptx` or `.cubin` |
| **Output naming** | `{module_id}.o` | `{module_id}.sm{arch}.ptx` or `.cubin` |

## How Warp Decides PTX vs CUBIN

From `Device.get_cuda_output_format()`:
1. If device arch not in `runtime.nvrtc_supported_archs` → PTX
2. User preference via `cuda_output` option → "ptx" or "cubin"
3. Default: PTX if driver_version >= toolkit_version, else CUBIN

## Changes Needed for GPU

1. **Test with real GPU**: The `extract_ir()` function already supports `device="cuda"` but needs GPU hardware to test
2. **Binary handling**: The `extract_ir()` correctly rejects `cubin` as non-text artifact
3. **Architecture suffix**: Path resolution uses `output_arch` for `.sm{arch}.ptx` files
4. **Fallback priority**: Current `prefer=("cpp", "cu", "ptx")` order is sensible - will find `.cpp` on CPU, `.cu`/`.ptx` on GPU

## Code Already in Place

```python
# ir_extractor.py already handles:
def _candidate_paths(module_id, module_dir, prefer, output_arch=None):
    # Generates paths like:
    # - "{module_dir}/{module_id}.cpp"
    # - "{module_dir}/{module_id}.cu"  
    # - "{module_dir}/{module_id}.sm{arch}.ptx"
    # - "{module_dir}/{module_id}.sm{arch}.cubin"
```

```python
# Architecture detection is in place:
output_arch = module.get_compile_arch(dev)  # Returns None for CPU, SM arch for CUDA
```

## New GPU-Specific Patterns to Add (if M3+ continues)

- [ ] Handle multiple SM architectures (same module compiled for different GPUs)
- [ ] Extract CUDA-specific metadata from `.meta` file (shared memory bytes, register usage)
- [ ] Add `prefer=("cu",)` option to get CUDA source before PTX for better readability
- [ ] Consider extracting intermediate NVRTC compilation logs if available

## Verification Checklist (for GPU environment)

```bash
# Test command for CUDA extraction (requires GPU)
python3 -c "
from jit.code.extraction.ir_extractor import extract_ir, extract_ir_artifact
from jit.code.extraction.test_ir_extractor import k_add, _compile_kernel
import warp as wp

wp.init()
_compile_kernel(k_add, device='cuda')

# Should return .cu file
artifact = extract_ir_artifact(k_add, device='cuda', prefer=('cu',))
print(f'Kind: {artifact.kind}, Path: {artifact.path}')

# Get CUDA source
ir = extract_ir(k_add, device='cuda', prefer=('cu',))
print(ir[:500])
"
```

## Summary

The `ir_extractor.py` implementation is **GPU-ready** in terms of code paths:
- Device parameter is supported
- CUDA file extensions (.cu, .ptx) are in the candidate list
- Architecture suffix handling is implemented for PTX/CUBIN paths
- Binary artifacts (cubin) are correctly rejected for text extraction

**Blocking issue**: No CUDA device available in this environment to verify end-to-end GPU path.
