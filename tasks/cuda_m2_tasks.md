# CUDA Milestone 2 Tasks

## Task 1: Test CUDA IR Extraction
- [ ] Step 1.1: Create test script to generate CUDA IR
- [ ] Step 1.2: Test ir_extractor with device="cuda" for simple kernel
- [ ] Step 1.3: Compare output structure vs CPU
- [ ] Step 1.4: Document any errors or issues (no GPU available)
- **Done when**: CUDA IR extraction code tested and output structure understood

## Task 2: Generate CUDA Samples
- [ ] Step 2.1: Modify pipeline to support device parameter
- [ ] Step 2.2: Generate 10+ CUDA sample pairs
- [ ] Step 2.3: Verify CUDA IR format and syntax
- [ ] Step 2.4: Save CUDA samples to data/cuda_samples/
- **Done when**: 10+ valid CUDA samples generated and saved

## Task 3: CPU vs CUDA Comparison
- [ ] Step 3.1: Create side-by-side comparison script
- [ ] Step 3.2: Compare function signatures
- [ ] Step 3.3: Compare thread ID handling
- [ ] Step 3.4: Compare memory operations
- [ ] Step 3.5: Compare type system differences
- **Done when**: notes/cuda_ir_format.md documents all major differences

## Task 4: Validate All Kernel Types
- [ ] Step 4.1: Test arithmetic kernels with CUDA
- [ ] Step 4.2: Test vector kernels with CUDA
- [ ] Step 4.3: Test matrix kernels with CUDA
- [ ] Step 4.4: Test control flow kernels with CUDA
- [ ] Step 4.5: Test math kernels with CUDA
- [ ] Step 4.6: Test atomic kernels with CUDA
- [ ] Step 4.7: Test remaining kernel types with CUDA
- **Done when**: All 10 kernel types generate valid CUDA IR
