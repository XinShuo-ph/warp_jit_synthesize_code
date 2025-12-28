# Chief Scientist Review: Warp JIT Codegen Datasets (CPU + CUDA)

## Executive summary
We generated two code datasets using **NVIDIA Warp**’s kernel **code generation**:
- **CPU dataset**: 200.06 MiB of Warp-generated **C++** kernel code
- **CUDA dataset**: 200.25 MiB of Warp-generated **CUDA C++** kernel code (generated without requiring a working GPU driver)

Both datasets are built from the same **10 kernel categories** and store, per record, the **Python kernel sources** plus the **full generated code** and **forward function extracts**.

## Background: JIT, IR, and Warp (why this matters)
- **JIT (Just-In-Time compilation)**: a compilation pipeline that generates low-level code at runtime from high-level source. In Warp, decorated `@wp.kernel` Python functions are turned into optimized backend-specific code (CPU/CUDA) on demand.
- **IR (Intermediate Representation)**: a compiler-internal representation between source and machine code. Practically, for Warp this can be viewed as the generated C++/CUDA source (and internal lowered forms) that capture the kernel’s semantics after type/lowering passes.
- **NVIDIA Warp**: a Python framework that compiles a restricted kernel DSL to efficient CPU/CUDA backends. Warp exposes a deterministic codegen path, making it well-suited for producing paired data: **Python kernel → generated backend code**.

## What we produced

### Dataset locations
- **CPU**: `data/production_cpu/cpu_code_dataset.jsonl`
- **CUDA**: `data/production_cuda/cuda_code_dataset.jsonl`
- **Manifests**:
  - `data/manifests/cpu_manifest.json`
  - `data/manifests/cuda_manifest.json`

### Schema (per JSONL record)
Each line is one JSON object with:
- **`python_sources`**: list of Warp `@wp.kernel` Python sources (default: 50 kernels per record)
- **`generated_code`**: full Warp-generated backend source for that module (`device=cpu` → C++; `device=cuda` → CUDA C++)
- **`kernels`**: per-kernel metadata (name, category, description, forward symbol)
- **`forward_functions`**: convenience extraction mapping `kernel_name → forward function code`
- **`warp_version`**, **`generated_at`**, **`module_source_hash`** for provenance

Generation entrypoint:
- `code/production/produce_warp_codegen_dataset.py`

### Size, counts, and timing
From the manifests:

**CPU**
- Size: **200.06 MiB** (`209,773,374` bytes)
- Records: **482**
- Kernels referenced: **24,100**
- Generation time: **46.4s**
- Warp version: **1.10.1**

**CUDA**
- Size: **200.25 MiB** (`209,978,937` bytes)
- Records: **491**
- Kernels referenced: **24,550**
- Generation time: **46.5s**
- Warp version: **1.10.1**

### Category coverage
Both datasets include kernels from these categories:
`arithmetic`, `vector`, `matrix`, `control_flow`, `math`, `atomic`, `nested_loop`, `multi_conditional`, `combined`, `scalar_param`.

The manifests contain per-category counts for the generated kernels (roughly uniform at ~2.4k/category per dataset).

## Notes / constraints
- This environment did **not** have a CUDA driver available; Warp initialized CPU-only, but **CUDA codegen still worked** (we generate `.cu` source without executing on GPU).
- Large dataset files are intentionally excluded from git by `.gitignore`; manifests and scripts are kept small and versionable.

