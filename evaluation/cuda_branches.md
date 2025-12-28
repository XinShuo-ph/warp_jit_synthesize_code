# CUDA Branch Evaluation

## Context
Per instruction_cuda.md: "Currently no GPU device is available for agent to use. Provide concise code and command for me to test on GPU device by myself."

## Key Finding
All branches (agent-work-merge-*) have CPU-based datasets. CUDA IR generation requires actual GPU hardware which is not available in this environment.

## Tested Branches

| Branch | JSON Files | Pipeline | CUDA Support | Recommendation |
|--------|-----------|----------|--------------|----------------|
| aa09   | 10,727    | ✅ Yes   | device param | ⭐ **SELECTED** |
| 81df   | 176       | ✅ Yes   | device param | Backup |
| 729a   | 100       | ✅ Yes   | device param | Backup |
| 9d9b   | 104       | ✅ Yes   | device param | Backup |
| 1496   | 30        | ✅ Yes   | device param | Backup |

## Selected: **branch aa09**

### Rationale
- **Most data**: 10,727 samples (same as top CPU branches)
- **CUDA-ready code**: Pipeline supports device parameter (cpu/cuda)
- **Proven quality**: Same codebase as successful CPU branches
- **Easy adaptation**: Just change device='cpu' to device='cuda'

### CUDA Dataset Strategy

Since CUDA IR compilation requires GPU hardware (not available), I will:

1. **Provide CUDA-adapted code** in `production/cuda/code/`:
   - Modified pipeline.py with device='cuda'
   - Modified generator.py with CUDA device specifications
   - Test suite for user to run on GPU

2. **Create CUDA-ready dataset structure** using CPU-generated samples as templates:
   - Same Python kernels work on both CPU and CUDA
   - Difference is only in IR compilation target
   - User can re-generate with actual CUDA device

### Key Code Changes for CUDA

```python
# In pipeline.py
def extract_ir_from_kernel(kernel, device: str = "cuda"):  # Changed from "cpu"
    # ... rest of code generates CUDA IR instead of CPU IR
    
# In main generation
device = "cuda"  # or "cuda:0" for specific device
```

### Files to Provide

1. `production/cuda/code/pipeline.py` - CUDA-enabled pipeline
2. `production/cuda/code/generator.py` - Same as CPU (kernels are device-agnostic)
3. `production/cuda/code/batch_generator.py` - CUDA batch generation
4. `production/cuda/test_on_gpu.sh` - Script for user to test on GPU
5. `production/cuda/README.md` - Instructions for GPU testing

### Expected Output on GPU

When user runs on actual GPU:
- IR will contain PTX/SASS code instead of CPU C++
- Function names: `*_cuda_kernel_forward` instead of `*_cpu_kernel_forward`
- CUDA-specific constructs: block/thread indices, shared memory, etc.

## Production Plan

Since actual CUDA generation requires GPU:

1. ✅ Copy CPU dataset structure as template (200MB)
2. ✅ Adapt code for CUDA device parameter
3. ✅ Create testing scripts for user
4. ✅ Document GPU requirements and testing procedure
5. ⏭️ User re-generates on GPU hardware for actual CUDA IR

## Note

The dataset I'm creating uses CPU IR as a placeholder. The Python kernels are valid CUDA kernels - only the IR compilation target differs. User must re-run generation on GPU to get actual CUDA IR (PTX/SASS).
