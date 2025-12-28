# Warp JIT CPU/CUDA Code Dataset (This Branch)

## Summary
This branch generates two code corpora for training/analysis:
- **CPU-generated code** (Warp `device="cpu"`): C++ source emitted by Warp’s codegen
- **CUDA-generated code** (Warp `device="cuda"`): CUDA C++ source emitted by Warp’s codegen

Both are produced from programmatically generated Warp kernels, saved as JSONL for downstream training pipelines.

## JIT (Just-In-Time compilation)
JIT compilation compiles code **at runtime** (or near runtime) instead of ahead-of-time. In Warp, Python kernels decorated with `@wp.kernel` are translated into lower-level code (CPU/CUDA), then compiled/cached as needed.

## IR (Intermediate Representation) in this project
“IR” here is used in the practical, pipeline sense: **an intermediate artifact between the high-level kernel description and executable code**. Concretely, we capture Warp’s **generated source** (C++ for CPU and CUDA C++ for CUDA) via Warp’s internal `ModuleBuilder.codegen(device)` API.

## NVIDIA Warp
NVIDIA Warp is a Python framework for writing high-performance kernels (CPU and GPU) with a single kernel language. Warp’s toolchain includes:
- A Python front-end (`@wp.kernel`)
- A code generation pipeline that can emit CPU/CUDA source
- A compilation/cache system for execution (GPU execution requires CUDA runtime/driver, but code generation does not)

## Current dataset (generated outputs)
Generated with Warp **1.10.1**.

### Files
- `jit/data/generated/cpu_code.jsonl`: **209,720,372 bytes (~200.0MB)**, **13,858** records
- `jit/data/generated/cuda_code.jsonl`: **209,720,027 bytes (~200.0MB)**, **13,246** records
- `jit/data/generated/dataset_stats.json`: run metadata + category distribution

### Record schema (JSONL)
Each line is a single JSON object:
```json
{
  "id": "kernelname_hash",
  "device": "cpu|cuda",
  "category": "arithmetic|vector|...",
  "description": "...",
  "arg_types": {"arg":"type", "...":"..."},
  "metadata": {"seed": 123, "...": "..."},
  "python_source": "@wp.kernel\\n...",
  "generated_code": "/* full generated C++/CUDA module text */",
  "warp_metadata": {"kernel_name":"...", "mangled_name":"...", "device":"...", "...":"..."}
}
```

### Kernel categories
The generator currently produces 10 categories:
`arithmetic`, `vector`, `matrix`, `control_flow`, `math`, `atomic`, `nested_loop`, `multi_conditional`, `combined`, `scalar_param`.

## How to reproduce
```bash
python3 -m pip install -r requirements.txt

# Produce both datasets (200MB each) + stats
python3 jit/code/synthesis/produce_codegen_dataset.py \
  --device both \
  --target-mb-cpu 200 \
  --target-mb-cuda 200 \
  --out-cpu jit/data/generated/cpu_code.jsonl \
  --out-cuda jit/data/generated/cuda_code.jsonl \
  --stats-out jit/data/generated/dataset_stats.json \
  --code-scope full
```

## Notes / limitations
- This environment has **no CUDA driver**, so Warp reports CPU-only mode, but **CUDA source code generation still succeeded** (we only require codegen text, not GPU execution).
- The generated datasets are large and are **ignored by git** by default via `.gitignore` (intended to avoid accidental commits of multi-hundred-MB artifacts).

