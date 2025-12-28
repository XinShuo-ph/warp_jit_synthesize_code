# Branch 3f34 Analysis

## Quick Stats
- **Milestone**: M2 (IR extraction) complete (per `README.md`)

## Unique Features
- **Debug tooling**:
  - `code/extraction/debug_loop.py` focuses on understanding loop lowering in IR.
  - `code/examples/check_codegen.py` / `check_install.py` are simple environment smoke checks.
  - `code/extraction/test_cuda_codegen.py` suggests early exploration of CUDA codegen paths.
- **Forward + backward extraction**: README emphasizes both forward and adjoint code extraction.

## Recommended for Merge
- [ ] Mine the debug scripts as optional developer utilities (especially loop IR inspection).
- [ ] Mine any CUDA codegen checks for future expansion (even if CI stays CPU-only).

## Skip / Handle Carefully
- This branch is only M2 (no synthesis pipeline / dataset generation).

