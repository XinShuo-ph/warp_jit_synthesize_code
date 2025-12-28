# CUDA Backend Validation

This directory contains the adapted code for generating and running Warp kernels on CUDA devices.

## Requirements

- NVIDIA GPU
- CUDA Toolkit installed
- Python 3.8+
- `warp-lang` installed

## Validation Steps

1. **Install Dependencies**
   ```bash
   pip install warp-lang numpy
   ```

2. **Run Validation Test**
   Execute the GPU execution test suite:
   ```bash
   python tests/test_cuda_execution.py
   ```
   
   If successful, it should verify:
   - CUDA device detection
   - Kernel compilation for CUDA
   - Kernel launch on GPU
   - Correctness of results (CPU <-> GPU memory transfer)

3. **Generate CUDA Training Data**
   Run the pipeline with the `--device cuda` flag:
   ```bash
   python code/synthesis/pipeline.py --device cuda -n 10
   ```
   
   Verify the output JSONs in `data/samples/` contain:
   - `"device": "cuda"` in metadata
   - CUDA C++ code (look for `blockDim`, `threadIdx`, etc.) in `cpp_forward`

## Troubleshooting

- **No CUDA device found**: Ensure `nvidia-smi` shows your GPU.
- **Compilation errors**: Check CUDA Toolkit version compatibility with Warp.
