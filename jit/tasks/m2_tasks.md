# Milestone 2 Tasks

## Task 1: IR Extractor Implementation
- [x] Step 1.1: Create ir_extractor.py with extract_ir(kernel) function
- [x] Step 1.2: Handle both CPU and CUDA code generation
- [x] Step 1.3: Return structured output (source, forward, backward)
- **Done when**: extract_ir() returns C++ code for any kernel ✓

## Task 2: Test IR Extraction
- [x] Step 2.1: Test with add_kernel (simple array ops)
- [x] Step 2.2: Test with dot_product (atomic ops)
- [x] Step 2.3: Test with saxpy (scalar + array)
- [x] Step 2.4: Test with branch_kernel (if/else)
- [x] Step 2.5: Test with loop_kernel (for)
- [x] Step 2.6: Test with vec_kernel (vector ops)
- **Done when**: 6 test cases produce valid IR pairs ✓

## Task 3: Document IR Format
- [x] Step 3.1: Analyze structure of generated C++ code
- [x] Step 3.2: Create notes/ir_format.md (max 30 lines)
- **Done when**: IR format is documented ✓
