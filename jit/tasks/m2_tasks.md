# Milestone 2 Tasks: IR Extraction Mechanism

## Task 1: Create IR Extractor Utility
- [x] Step 1.1: Create `ir_extractor.py` with `extract_ir(kernel)` function
- [x] Step 1.2: Handle both CPU and CUDA code generation
- [x] Step 1.3: Return structured dict with {python_source, cpp_code, metadata}
- **Done when**: Function successfully extracts IR from any valid warp kernel ✓

## Task 2: Test with Multiple Kernels
- [x] Step 2.1: Test with simple arithmetic kernel
- [x] Step 2.2: Test with vector operations kernel
- [x] Step 2.3: Test with matrix operations kernel
- [x] Step 2.4: Test with control flow kernel (if/for)
- [x] Step 2.5: Test with warp built-in functions
- **Done when**: 5+ kernels successfully produce Python→IR pairs ✓ (7/7)

## Task 3: Document IR Format
- [x] Step 3.1: Document the structure of generated C++ code
- [x] Step 3.2: Note key patterns (var naming, forward/backward structure)
- **Done when**: notes/ir_format.md exists with <30 lines ✓

## Task 4: Create Sample Pairs
- [x] Step 4.1: Save 5+ sample pairs to data/ directory
- **Done when**: data/ contains validated Python→IR pair files ✓
