# Milestone 4 Tasks

## Task 1: Design kernel generation strategy
- [x] Step 1.1: Define kernel templates (operations, patterns)
- [x] Step 1.2: Define parameter spaces (array sizes, types, values)
- [x] Step 1.3: Create variation strategies (simple→complex)
- [x] Step 1.4: Document generation approach
- **Done when**: Clear strategy for generating diverse kernels

## Task 2: Implement kernel generator
- [x] Step 2.1: Create KernelGenerator class
- [x] Step 2.2: Implement template-based generation
- [x] Step 2.3: Add randomization with seeds for reproducibility
- [x] Step 2.4: Test generator produces valid kernels
- **Done when**: Can generate 10+ varied kernels programmatically

## Task 3: Build end-to-end pipeline
- [x] Step 3.1: Integrate generator with IR extractor
- [x] Step 3.2: Add compilation and validation steps
- [x] Step 3.3: Implement data saving (JSON format)
- [x] Step 3.4: Add progress tracking and error recovery
- **Done when**: Pipeline runs from generation → IR extraction → save

## Task 4: Generate initial dataset
- [x] Step 4.1: Run pipeline to generate 100+ samples
- [x] Step 4.2: Validate all samples
- [x] Step 4.3: Check for diversity (no duplicates)
- [x] Step 4.4: Verify data format consistency
- **Done when**: 100+ valid Python→IR pairs in data/samples/

## Task 5: Test and document
- [x] Step 5.1: Run pipeline twice, verify reproducibility
- [x] Step 5.2: Test with different random seeds
- [x] Step 5.3: Document pipeline usage
- [x] Step 5.4: Create example scripts
- **Done when**: Pipeline is documented and reproducible
