# Milestone 1 Tasks: Environment Setup & Warp Basics

## Task 1: Environment Setup
- [x] Step 1.1: Install `warp-lang` and dependencies.
- [x] Step 1.2: Clone NVIDIA/warp repository for reference (e.g. into `warp_repo`).
- [x] Step 1.3: Verify installation by importing warp in python.
- **Done when**: `import warp; warp.init()` runs without error in a python script.

## Task 2: Run Examples
- [x] Step 2.1: Locate a simple example in `warp_repo/examples` (e.g., `example_sim_particles.py` or similar simple one).
- [x] Step 2.2: Copy it to `code/examples/` and run it.
- [x] Step 2.3: Verify output.
- **Done when**: At least one example runs successfully and produces expected output.

## Task 3: Understand Kernel Compilation
- [x] Step 3.1: Read `warp/codegen.py` and `warp/context.py` in the cloned repo.
- [x] Step 3.2: Write a minimal script to define a kernel and inspect its generated code (if possible) or just run it.
- [x] Step 3.3: Document findings in `notes/warp_basics.md`.
- **Done when**: `notes/warp_basics.md` exists and explains how kernels compile and where IR might be found.
