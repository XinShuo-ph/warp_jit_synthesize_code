# Warp JIT Code Synthesis - cursor/instructions-wrapup-completion-fc9f

## Progress Summary
- **Milestone reached**: M3 (completed)
- **Key deliverables**:
  - IR extraction mechanism for Warp kernels (Python → C++ source)
  - 5 sample Python→IR pairs in JSONL format
  - Warp FEM Poisson solver example with validation tests
  - Technical documentation on Warp compilation and IR format

## What Works
- **IR Extraction** (`ir_extractor.py`): Extracts generated C++ source code from Warp kernels using `ModuleBuilder.codegen()`. Works for CPU target.
- **Sample Generation** (`m2_generate_pairs.py`): Generates deterministic Python→IR training pairs (5 kernels: add, saxpy, clamp, where, sin_cos).
- **Poisson Solver** (`poisson_solver.py`): Solves -Δu = f on [0,1]² with Dirichlet BC using Warp FEM. Supports grid/tri/quad meshes.
- **Validation Tests** (`test_poisson.py`): Verifies error decreases with mesh refinement.
- **Example Kernels**: Basic Warp kernel examples (add, saxpy, reduction) demonstrating JIT compilation.

## Requirements
```bash
pip install warp-lang
```

## Quick Start
```bash
# Set PYTHONPATH to workspace root
export PYTHONPATH=/workspace

# Run IR extraction and generate pairs
python3 jit/code/extraction/m2_generate_pairs.py

# Run Poisson solver example
python3 jit/code/examples/poisson_solver.py

# Run Poisson validation test
python3 jit/code/examples/test_poisson.py

# Run basic kernel examples
python3 jit/code/examples/ex00_add.py
python3 jit/code/examples/ex01_saxpy.py
python3 jit/code/examples/ex02_reduction.py
```

## File Structure
```
jit/
├── code/
│   ├── extraction/          # IR extraction utilities
│   │   ├── ir_extractor.py  # Core extraction: kernel → C++ source
│   │   └── m2_generate_pairs.py  # Generate Python→IR sample pairs
│   └── examples/            # Example kernels and FEM solver
│       ├── ex00_add.py      # Vector addition kernel
│       ├── ex00_smoke.py    # Smoke test
│       ├── ex01_saxpy.py    # SAXPY kernel
│       ├── ex02_reduction.py # Reduction kernel
│       ├── poisson_solver.py # Warp FEM Poisson solver
│       └── test_poisson.py  # Validation tests
├── data/
│   └── samples/
│       └── m2_pairs.jsonl   # 5 Python→IR training pairs
├── notes/
│   ├── ir_format.md         # Documentation of IR format
│   └── warp_basics.md       # Warp compilation basics
├── tasks/
│   ├── m1_tasks.md          # M1 task checklist (completed)
│   ├── m2_tasks.md          # M2 task checklist (completed)
│   └── m3_tasks.md          # M3 task checklist (completed)
├── README.md                # This file
├── WRAPUP_STATE.md          # Wrapup progress tracker
└── instructions_wrapup.md   # Wrapup instructions (read-only)
```

## Generated Data Format
Each line in `m2_pairs.jsonl` is a JSON object:
```json
{
  "name": "k_add",
  "python": "@wp.kernel(module=\"unique\")\ndef k_add(a: wp.array(dtype=wp.float32), ...):\n    ...",
  "ir": "#define WP_TILE_BLOCK_DIM 256\n#include \"builtin.h\"\n...\nvoid k_add_cpu_kernel_forward(...) {...}",
  "meta": {
    "warp_version": "1.10.1",
    "device": "cpu",
    "codegen_device": "cpu",
    "mangled_name": "k_add_efafbfce",
    "module_name": "k_add_70ebb21e",
    "module_hash": "70ebb21e..."
  }
}
```

## Known Issues / TODOs
- **CPU-only**: All validation done in CPU mode (no CUDA driver in environment)
- **GPU support**: `ir_extractor.py` has `device` parameter but CUDA codegen untested
- **M4 not started**: Synthesis pipeline (`code/synthesis/`) not yet implemented
- **Limited sample count**: Only 5 training pairs generated (≤100 recommended for git)
