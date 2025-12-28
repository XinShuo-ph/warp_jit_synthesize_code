# CUDA Milestone 4 Tasks - Batch Generation & Validation

## Task 1: Enhanced Pipeline with Backward Support
- [ ] Step 1.1: Modify pipeline.py to support include_backward parameter
- [ ] Step 1.2: Update batch_generator.py for CUDA device support
- [ ] Step 1.3: Test batch generation with forward+backward
- [ ] Step 1.4: Generate 100+ samples with both passes
- **Done when**: Batch pipeline generates forward+backward CUDA samples

## Task 2: Test Suite for GPU Validation
- [ ] Step 2.1: Create test_cuda_kernels.py with all kernel type tests
- [ ] Step 2.2: Add runtime tests (array creation, kernel launch)
- [ ] Step 2.3: Add correctness tests (compare outputs)
- [ ] Step 2.4: Document test execution on GPU
- **Done when**: Comprehensive test suite ready for GPU execution

## Task 3: GPU Execution Scripts
- [ ] Step 3.1: Create run_on_gpu.sh script
- [ ] Step 3.2: Add environment check (CUDA availability)
- [ ] Step 3.3: Add batch test execution
- [ ] Step 3.4: Add result validation and reporting
- **Done when**: User can run `./tests/run_on_gpu.sh` to validate all

## Task 4: Documentation for GPU Testing
- [ ] Step 4.1: Create CUDA_TESTING.md with prerequisites
- [ ] Step 4.2: Document expected outputs
- [ ] Step 4.3: Document troubleshooting steps
- [ ] Step 4.4: Create example test runs with expected results
- **Done when**: Clear instructions for user to test on GPU hardware

## Task 5: Sample Data Curation
- [ ] Step 5.1: Generate diverse samples (100+)
- [ ] Step 5.2: Verify JSON format consistency
- [ ] Step 5.3: Create train/validation split
- [ ] Step 5.4: Commit ≤100 samples to git (rest in .gitignore)
- **Done when**: Sample data ready for LLM training
