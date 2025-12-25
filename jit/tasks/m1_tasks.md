# Milestone 1 Tasks

## Task 1: Repo scaffolding + M1 plan
- [x] Step 1.1: Create `jit/` directory structure from `instructions.md`
- [x] Step 1.2: Create `jit/STATE.md` and set next action
- [x] Step 1.3: Create `jit/tasks/m1_tasks.md` with testable "Done when"
- **Done when**: `jit/STATE.md` and this task file exist, and Task 2 has concrete run commands.

## Task 2: Install Warp and validate import
- [x] Step 2.1: Install `warp-lang` via pip
- [x] Step 2.2: Verify `import warp as wp` works and print `wp.__version__`
- [x] Step 2.3: Record whether CUDA is available via `wp.is_cuda_available()`
- **Done when**: A one-liner Python command succeeds twice and reports version + CUDA availability.

## Task 3: Run 3+ Warp kernel examples (CPU OK)
- [x] Step 3.1: Add `jit/code/examples/add.py` (vector add kernel) and run twice
- [x] Step 3.2: Add `jit/code/examples/saxpy.py` (saxpy kernel) and run twice
- [x] Step 3.3: Add `jit/code/examples/reduction_sum.py` (simple reduction) and run twice
- [x] Step 3.4: Ensure examples run without debug prints and exit 0
- **Done when**: All 3 scripts complete successfully twice with consistent numeric outputs.

## Task 4: Document Warp compilation basics (max 50 lines)
- [x] Step 4.1: Write `jit/notes/warp_basics.md` (<=50 lines)
- [x] Step 4.2: Include: kernel definition → build/compile → module caching; where to look for IR hooks in Warp source
- **Done when**: File exists, <=50 lines, and references the key Warp modules to study later (codegen/context).

