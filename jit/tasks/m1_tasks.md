# Milestone 1 Tasks

## Task 1: Environment Setup
- [x] Step 1.1: Install warp-lang package
- [x] Step 1.2: Clone warp repo for examples
- [x] Step 1.3: Verify warp initializes correctly
- **Done when**: `import warp; warp.init()` succeeds

## Task 2: Run Basic Examples
- [x] Step 2.1: Create vector addition kernel (ex1_basic_kernel.py)
- [x] Step 2.2: Create math operations kernel (ex2_math_ops.py)  
- [x] Step 2.3: Create vector types kernel (ex3_vec_types.py)
- [x] Step 2.4: Verify all examples pass twice
- **Done when**: All 3 examples output SUCCESS on 2 consecutive runs

## Task 3: Understand Compilation Flow
- [x] Step 3.1: Explore kernel object attributes
- [x] Step 3.2: Locate generated C++ in cache (~/.cache/warp/)
- [x] Step 3.3: Create IR extraction utility (ir_extractor.py)
- **Done when**: Can extract Python source and generated C++ from kernel

## Task 4: Documentation
- [x] Step 4.1: Create notes/warp_basics.md (max 50 lines)
- **Done when**: warp_basics.md exists with kernel compilation flow documented
