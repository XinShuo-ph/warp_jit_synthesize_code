# Milestone 1 Tasks

## Task 1: Environment Setup
- [x] Step 1.1: Create directory structure (jit/, tasks/, code/, data/, notes/)
- [x] Step 1.2: Install warp-lang package
- [x] Step 1.3: Clone warp repo for examples reference
- [x] Step 1.4: Verify warp initializes correctly
- **Done when**: `import warp as wp; wp.init()` succeeds

## Task 2: Run Basic Examples
- [x] Step 2.1: Run test_add_kernel.py - simple array add
- [x] Step 2.2: Run test_dot_product.py - atomic operations
- [x] Step 2.3: Run test_saxpy.py - scalar+array ops
- **Done when**: 3 examples run without errors ✓

## Task 3: Understand Kernel Compilation Flow
- [x] Step 3.1: Study codegen.py to understand how kernels are compiled
- [x] Step 3.2: Find where IR is generated/stored (~/.cache/warp/)
- [x] Step 3.3: Create notes/warp_basics.md with findings (max 50 lines)
- **Done when**: Can explain: Python kernel → IR → compiled code flow ✓
