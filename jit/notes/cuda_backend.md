# CUDA Backend Notes (Warp JIT Synthesis)

## What “CUDA backend” means here
We generate and extract **Warp’s generated source** for a target backend:
- **CPU** → C++ (`.cpp`)
- **CUDA** → CUDA C/C++ (`.cu`)

The extractor/pipeline selects the backend via a `device` string (`cpu` or `cuda`).

## Device detection gotcha
`warp.get_devices()` may return a list of `Device` objects (not plain strings). Use `d.alias` (or string fallback) to check availability.

## Extraction strategy
- Use `warp._src.context.ModuleBuilder(...).codegen(device)` to get the full generated module source
- Extract a single kernel function body by name: `<mangled>_<device>_kernel_forward` (and `_kernel_backward` when enabled)
- CUDA code may prefix functions with qualifiers like `__global__`, so function matching must allow common qualifiers.

## How to validate on a GPU machine

```bash
pip install -U warp-lang pytest
python3 -m pytest -q
python3 jit/code/synthesis/pipeline.py -n 5 -o /tmp/jit_cuda_smoke --device cuda
```
