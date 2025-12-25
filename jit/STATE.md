# Current State
- **Milestone**: M2
- **Task**: M2 complete (kernel-cache IR extraction + 6 kernel→IR cases)
- **Status**: ready_for_next

## Next Action
Create `jit/tasks/m3_tasks.md`, then start M3 by prototyping `jit/code/examples/poisson_solver.py` (Warp FEM Poisson solve) and a small validation in `jit/code/examples/test_poisson.py`.

## Blockers
None

## Session Log
- 2025-12-25: Created `jit/` structure + `jit/tasks/m1_tasks.md`; installed `warp-lang` (CPU-only here); ran 3 examples twice each; wrote `jit/notes/warp_basics.md`.
- 2025-12-25: Implemented `jit/code/extraction/ir_extractor.py` (extract cached `.cpp/.cu/.ptx`); added `jit/code/extraction/test_ir_extractor.py` with 6 kernels; ran tests twice; wrote `jit/notes/ir_format.md`.

