# Milestone 1 Tasks (Environment Setup & Warp Basics)

## Task 1: Scaffold project workspace
- [x] Step 1.1: Create `jit/` folders (tasks/code/data/notes)
- [x] Step 1.2: Ensure `STATE.md` references `jit/` paths for deliverables
- **Done when**: `jit/tasks/`, `jit/code/`, `jit/data/`, `jit/notes/` exist and `STATE.md` next action points into `jit/`.

## Task 2: Install Warp and confirm import
- [x] Step 2.1: Install `warp-lang` with `python3 -m pip install -U warp-lang`
- [x] Step 2.2: Run a tiny kernel on CPU to confirm JIT works
- **Done when**: `python3 -c "import warp as wp; print(wp.__version__)"` works and a kernel launch completes.

## Task 3: Run 3+ examples successfully
- [x] Step 3.1: Create and run `jit/code/examples/ex00_add.py`
- [x] Step 3.2: Create and run `jit/code/examples/ex01_saxpy.py`
- [x] Step 3.3: Create and run `jit/code/examples/ex02_reduction.py`
- **Done when**: All three scripts run end-to-end twice with identical outputs.

## Task 4: Write minimal compilation notes
- [x] Step 4.1: Record how a kernel is defined/compiled/launched
- [x] Step 4.2: Identify where Warp stores/prints intermediate artifacts (as available)
- **Done when**: `jit/notes/warp_basics.md` exists and is <= 50 lines.

