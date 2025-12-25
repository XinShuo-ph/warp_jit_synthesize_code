# Milestone 1 Tasks

## Task 1: Environment Setup
- [x] Step 1.1: Install warp-lang package
- [x] Step 1.2: Create directory structure (code/, data/, notes/, tasks/)
- [x] Step 1.3: Clone warp repository for reference
- **Done when**: `import warp as wp; wp.init()` runs without errors

## Task 2: Run Warp Examples
- [x] Step 2.1: Create and run basic array addition kernel
- [x] Step 2.2: Create and run vector operations kernel  
- [x] Step 2.3: Create and run kernel with functions and control flow
- **Done when**: 3+ different kernels execute successfully

## Task 3: Understand Kernel Compilation
- [x] Step 3.1: Locate kernel cache directory (~/.cache/warp/)
- [x] Step 3.2: Examine generated C++ code structure
- [x] Step 3.3: Identify key components (forward/backward functions, SSA vars)
- **Done when**: Can manually find and read generated .cpp files

## Task 4: IR Extraction Mechanism
- [x] Step 4.1: Create function to programmatically access cached .cpp files
- [x] Step 4.2: Test extraction on compiled kernels
- [x] Step 4.3: Verify IR contains source line mappings
- **Done when**: Python function returns C++ code for any kernel

## Task 5: Documentation
- [x] Step 5.1: Document compilation flow (AST→SSA→C++→binary)
- [x] Step 5.2: Document cache structure and naming conventions
- [x] Step 5.3: Document key warp source files
- **Done when**: notes/warp_basics.md exists and is ≤50 lines

## Validation
- [x] All 3+ examples run successfully twice with consistent results
- [x] IR extraction works programmatically
- [x] Documentation complete
- [x] No debug code or uncommitted files

## Milestone 1 Complete ✓
All tasks completed successfully.
