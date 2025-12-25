# Milestone 1 Tasks

## Task 1: Environment setup (warp + runtime deps)
- [x] Step 1.1: Confirm Python is available (`python --version`) and pip works.
- [x] Step 1.2: Install Warp (`pip install -U warp-lang`).
- [x] Step 1.3: In a fresh Python process, import Warp and print version (`python -c "import warp as wp; print(wp.__version__)"`).
- **Done when**: Warp imports successfully and prints a version string.

## Task 2: Run 3+ Warp examples successfully
- [x] Step 2.1: Locate installed Warp examples directory (via `python -c "import warp, os; print(os.path.dirname(warp.__file__))"`).
- [x] Step 2.2: Run example #1 twice from clean shell (same command both times) and confirm it completes without error. (`examples/core/example_graph_capture.py --device cpu --headless --num_frames 2`)
- [x] Step 2.3: Run example #2 twice from clean shell and confirm it completes without error. (`examples/tile/example_tile_matmul.py --device cpu --headless`)
- [x] Step 2.4: Run example #3 twice from clean shell and confirm it completes without error. (`examples/tile/example_tile_cholesky.py --device cpu --headless`)
- **Done when**: 3 distinct examples complete successfully twice each.

## Task 3: Document Warp kernel compilation flow (brief)
- [x] Step 3.1: Identify how a `wp.kernel` is compiled (entry points to study: `warp/context.py`, `warp/codegen.py`).
- [x] Step 3.2: Write `notes/warp_basics.md` (≤ 50 lines) describing: compilation trigger, where generated code/IR lives, and how to locate it programmatically.
- **Done when**: `notes/warp_basics.md` exists, ≤ 50 lines, and is actionable (points to exact modules/functions).

