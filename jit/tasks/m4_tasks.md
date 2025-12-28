# Milestone 4 Tasks: Synthesis Pipeline

## Task 1: Create Kernel Generator
- [x] Step 1.1: Create generator.py with kernel template system
- [x] Step 1.2: Implement variations: arithmetic, vector, matrix, control flow
- [x] Step 1.3: Add randomization for kernel parameters
- **Done when**: Can programmatically generate diverse valid warp kernels ✓

## Task 2: Create End-to-End Pipeline
- [x] Step 2.1: Create pipeline.py that: generate → compile → extract → save
- [x] Step 2.2: Handle compilation errors gracefully
- [x] Step 2.3: Save pairs as JSON with metadata
- **Done when**: Pipeline produces validated Python→IR pairs ✓

## Task 3: Generate Sample Dataset
- [x] Step 3.1: Generate 100+ varied kernel pairs
- [x] Step 3.2: Validate all pairs compile correctly (120/120)
- [x] Step 3.3: Save to data/samples/ directory
- **Done when**: data/samples/ contains 100+ validated pairs ✓ (125 total)

## Kernel Variation Categories
1. Arithmetic: +, -, *, /, mix of operations
2. Vector: dot, cross, normalize, length
3. Matrix: mat*vec, mat*mat, transpose
4. Control flow: if/elif/else, for loops
5. Math functions: sin, cos, exp, sqrt, etc.
6. Atomics: atomic_add, atomic_min, etc.
