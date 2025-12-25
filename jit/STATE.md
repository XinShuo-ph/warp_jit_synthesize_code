# Current State
- **Milestone**: M2
- **Task**: IR extraction mechanism (CPU codegen as IR)
- **Status**: ready_for_next

## Next Action
Start M3: create `jit/tasks/m3_tasks.md`, then add `jit/code/examples/poisson_solver.py` using `warp.fem` and a small validation test `jit/code/examples/test_poisson.py` (run twice).

## Blockers (if any)
None

## Session Log
- 2025-12-25: Created `jit/` directory structure, installed `warp-lang` (CPU-only), added 3 runnable kernels, and wrote `notes/warp_basics.md`.
- 2025-12-25: Implemented CPU "IR" extraction via `ModuleBuilder.codegen()`, added 5 deterministic Python→IR test cases, and documented IR format.

## Notes
- Per environment policy, no git commits were created in this session.

