# Current State
- **Milestone**: M1
- **Task**: M1 complete (Warp installed + examples run + notes written)
- **Status**: ready_for_next

## Next Action
Create `jit/tasks/m2_tasks.md`, then begin M2 by implementing `jit/code/extraction/ir_extractor.py` to return cached generated source (`.cpp`/`.cu`) for a given compiled kernel (using `warp._src.context.get_module(...).get_module_identifier()` + `warp.config.kernel_cache_dir`).

## Blockers
None

## Session Log
- 2025-12-25: Created `jit/` structure + `jit/tasks/m1_tasks.md`; installed `warp-lang` (CPU-only here); ran 3 examples twice each; wrote `jit/notes/warp_basics.md`.

