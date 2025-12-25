# Current State
- **Milestone**: M2
- **Task**: M2 complete (Python kernel -> generated C++ source extraction)
- **Status**: ready_for_next

## Next Action
1. Create `jit/tasks/m3_tasks.md` with a concrete breakdown for the FEM Poisson solver milestone (M3).
2. Implement `jit/code/examples/poisson_solver.py` using `warp.fem` (solve Poisson with an analytic solution for validation).
3. Add `jit/code/examples/test_poisson.py` and run tests twice consecutively.

## Blockers
None

## Session Log
- 2025-12-25: Created `jit/` layout, added M1 tasks, installed `warp-lang==1.10.1` (CPU-only), cloned Warp repo to `jit/_deps/warp` (git-ignored).
  - Verified init: `python3 -c "import warp as wp; wp.init(); print(wp.__version__)"` (kernel cache printed at `/home/ubuntu/.cache/warp/1.10.1`).
  - Ran 3 examples twice each (CPU/headless):
    - `warp/examples/core/example_graph_capture.py --device cpu --headless --num_frames 2`
    - `warp/examples/fem/example_diffusion.py --device cpu --headless --quiet --resolution 10`
    - `warp/examples/fem/example_burgers.py --device cpu --headless --quiet --resolution 10 --num_frames 2`
  - Wrote `jit/notes/warp_basics.md` (<= 50 lines) and pinned `jit/requirements.txt`.
- 2025-12-25: Completed M2 IR extraction mechanism on CPU:
  - Implemented `jit/code/extraction/ir_extractor.py` (`extract_ir()` returns Warp-generated CPU C++ source via `ModuleBuilder.codegen("cpu")`).
  - Added fixtures in `jit/code/extraction/fixture_kernels.py` and 5 stable pytest cases in `jit/code/extraction/test_ir_extractor.py` (tests pass twice).
  - Documented chosen IR format in `jit/notes/ir_format.md` (<= 30 lines).

