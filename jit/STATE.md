# Current State
- **Milestone**: M1
- **Task**: M1 complete (environment + basics)
- **Status**: ready_for_next

## Next Action
Start M2: create `jit/tasks/m2_tasks.md`, then inspect the installed Warp package to identify the programmatic hook(s) to capture kernel IR during/after compilation (likely around `warp/context.py` + `warp/codegen.py`), and add `jit/code/extraction/ir_extractor.py` skeleton.

## Blockers (if any)
None

## Session Log
- 2025-12-25: Created `jit/` directory structure, installed `warp-lang` (CPU-only), added 3 runnable kernels, and wrote `notes/warp_basics.md`.

