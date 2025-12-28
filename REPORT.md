## Warp JIT Code Dataset (CPU + CUDA) — Brief Report

### Executive summary
We generated two code datasets using **NVIDIA Warp**’s JIT code generator:
- **CPU dataset**: **213,476,688 bytes (~203.6MB)**, **25,500 records** (`artifacts/datasets/cpu/dataset.jsonl`)
- **CUDA dataset**: **210,335,832 bytes (~200.6MB)**, **43,800 records** (`artifacts/datasets/cuda/dataset.jsonl`)

Each record pairs **Python Warp kernel source** with the corresponding **generated backend code** (forward + backward/adjoint when available).

---

### What “JIT” means here
**Just-in-time (JIT) compilation** compiles code at runtime, after seeing concrete kernel definitions and (often) runtime configuration such as target backend (`cpu` vs `cuda`). In Warp, Python kernels decorated with `@wp.kernel` are analyzed, transformed, and lowered into backend code when code generation is requested.

Why this matters for training data: JIT systems expose a rich mapping from **high-level intent** (Python kernel) to **lower-level implementation** (backend code). That mapping is directly useful for training models to reason about compilation, kernel lowering, and performance-relevant transformations.

---

### What “IR” means in this project
Strictly speaking, an “IR” is an internal representation used inside a compiler (e.g., SSA graphs, bytecode, LLVM IR). In this dataset, we use Warp’s **generated backend source** as a practical, stable *IR proxy*:
- CPU backend: Warp-generated **C/C++** kernel functions
- CUDA backend: Warp-generated **CUDA C/C++** kernel functions

This is valuable because it is explicit, readable, and captures backend-specific lowering decisions (thread indexing, memory access patterns, vector/matrix lowering, etc.).

---

### NVIDIA Warp (why it’s used)
**NVIDIA Warp** is a Python framework for writing high-performance simulation/geometry kernels that can target **CPU** and **GPU** backends. It provides:
- A kernel programming model (`@wp.kernel`, `wp.tid()`, typed arrays/vectors/matrices)
- Automatic differentiation (forward/backward/adjoint kernels)
- A compilation pipeline that can emit backend code for CPU and CUDA

These properties make it a good engine for synthesizing diverse, structured training pairs.

---

### Current dataset: format and coverage

#### Storage format
Both datasets are **JSONL** (one JSON object per line) with the core fields:
- `python_source`: the Warp kernel source (string)
- `code_forward`: generated backend function for the forward pass (string)
- `code_backward`: generated backend function for the backward/adjoint pass (string, may be `null` if unavailable)
- `device`: `"cpu"` or `"cuda"`
- `kernel_type`: category label used during synthesis
- `metadata.seed`: deterministic seed used for generation

#### CPU dataset composition
- **Records**: 25,500
- **Types** (balanced): `arithmetic`, `conditional`, `loop`, `math`, `vector`, `atomic`, `nested`, `multi_cond`, `combined`, `scalar_param`

#### CUDA dataset composition
- **Records**: 43,800
- **Types** (balanced): `arithmetic`, `math`, `vector`, `matrix`, `control_flow`, `atomic`

#### Notes on CUDA environment
Generation was performed on a machine without an NVIDIA driver (Warp ran in CPU-only mode), but the dataset contains **CUDA backend code** produced via Warp’s code generator. Executing these kernels requires a GPU-capable environment, but code generation (and thus dataset creation) does not.

---

### Reproducibility
The size-targeted generator script is:
- `tools/generate_dataset_to_size.py`

It generates until `dataset.jsonl` reaches the requested byte target and writes a matching `stats.json` alongside.

