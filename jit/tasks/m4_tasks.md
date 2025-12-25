# Milestone 4 Tasks

## Task 1: Kernel Generator
- [x] Step 1.1: Define kernel templates (arithmetic, control flow, loops, vectors)
- [x] Step 1.2: Implement random variation (operations, types, array sizes)
- [x] Step 1.3: Generate valid kernel functions dynamically
- **Done when**: generator.py can create 10 different valid kernels

## Task 2: Synthesis Pipeline
- [x] Step 2.1: Create pipeline that generates kernel → compiles → extracts IR
- [x] Step 2.2: Handle compilation errors gracefully
- [x] Step 2.3: Save pairs as JSON files with metadata
- **Done when**: pipeline.py processes kernels end-to-end

## Task 3: Sample Generation
- [x] Step 3.1: Generate 100+ varied kernels
- [x] Step 3.2: Validate all pairs have valid Python and C++ IR
- [x] Step 3.3: Store in data/samples/
- **Done when**: 100+ valid pairs in data/samples/ (120 generated)
