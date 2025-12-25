# Current State
- **Milestone**: M2
- **Task**: Completed (codegen IR extraction + 5 sample pairs)
- **Status**: ready_for_next

## Next Action
1. Create `jit/tasks/m3_tasks.md` (warp.fem Poisson solver) with testable steps and “Done when”.
2. Implement `jit/code/examples/poisson_solver.py` and `jit/code/examples/test_poisson.py`.
3. Run the Poisson tests twice with identical results.

## Blockers
None

## Session Log
- (initial): Project initialized, ready to begin M1
- (2025-12-25): Completed M1 in CPU-only mode: installed `warp-lang` 1.10.1, ran 3 example scripts twice each, and wrote `jit/notes/warp_basics.md`.
- (2025-12-25): Completed M2: added `jit/code/extraction/ir_extractor.py`, generated `jit/data/samples/m2_pairs.jsonl` (5 Python→IR pairs via `ModuleBuilder.codegen()`), and wrote `jit/notes/ir_format.md`.

