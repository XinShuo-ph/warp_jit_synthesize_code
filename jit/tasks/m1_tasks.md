# Milestone 1 Tasks

## Task 1: Environment Setup
- [x] Step 1.1: Install warp-lang package
- [x] Step 1.2: Clone warp repo for reference
- [x] Step 1.3: Create folder structure
- **Done when**: `import warp as wp; wp.init()` works

## Task 2: Run Warp Examples
- [x] Step 2.1: Create and run basic kernel tests (add, saxpy, dot)
- [x] Step 2.2: Run example_dem.py
- [x] Step 2.3: Run example_diffusion.py (FEM)
- **Done when**: 3+ examples run without errors

## Task 3: Understand Kernel Compilation
- [x] Step 3.1: Explore Kernel and Adjoint class attributes
- [x] Step 3.2: Locate generated C++ code in cache
- [x] Step 3.3: Document IR format and extraction path
- **Done when**: Can read Python source and C++ IR from kernel

## Task 4: Document Findings
- [x] Step 4.1: Create notes/warp_basics.md (max 50 lines)
- **Done when**: Notes explain compilation flow and IR access
