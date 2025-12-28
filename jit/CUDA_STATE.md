# CUDA State
- **Milestone**: M5
- **Task**: CUDA backend implementation complete (pending GPU validation)
- **Status**: ready_for_next

## Next Action
On a GPU machine:
1. `pip install -U warp-lang pytest`
2. `python3 -m pytest -q` (CUDA test should run; CPU-only env will skip it)
3. `python3 jit/code/synthesis/pipeline.py -n 5 -o /tmp/jit_cuda_smoke --device cuda`
4. Optionally: `python3 jit/code/synthesis/batch_generator.py -n 100 -o /tmp/jit_cuda_batch --device cuda`

## Blockers (if any)
- No GPU available in this environment; CUDA tests must be runtime-skipped.

## Session Log
- 2025-12-28:
  - Imported `jit/` baseline from `following-instructions-md-12c4`, pruned large datasets and caches, added `.gitignore`.
  - Implemented `device=cpu|cuda` extraction and pipeline/batch support with runtime CUDA detection.
  - Added pytest CUDA smoke test (skips if CUDA unavailable) and updated docs/runbook.
