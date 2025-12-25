# Milestone 1 Tasks: Environment Setup & Warp Basics

## Task 1: Environment Setup
- [x] Step 1.1: Install warp-lang package
- [x] Step 1.2: Clone warp repo for examples reference
- [x] Step 1.3: Verify warp imports and initializes correctly
- **Done when**: `import warp as wp; wp.init()` runs without error

## Task 2: Run Warp Examples
- [x] Step 2.1: Run array addition kernel (test_basic_kernels.py)
- [x] Step 2.2: Run sine wave computation kernel
- [x] Step 2.3: Run vector normalization kernel
- **Done when**: 3 examples run without errors and produce expected output ✓

## Task 3: Understand Kernel Compilation Flow
- [x] Step 3.1: Study warp/codegen.py for IR generation
- [x] Step 3.2: Study warp/context.py for kernel compilation
- [x] Step 3.3: Identify how to access generated IR/C++ from compiled kernel
- **Done when**: Can explain where IR lives and how to access it ✓

## Task 4: Document Findings
- [x] Step 4.1: Create notes/warp_basics.md with compilation flow summary
- **Done when**: File exists with <50 lines explaining kernel compile -> IR flow ✓
