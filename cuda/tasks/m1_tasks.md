# Milestone 1 Tasks: Baseline Setup & Analysis

## Task 1.1: Select and Copy Base Branch
- [x] Review branch_progresses.md and select 12c4 as base
- [ ] Copy key files from 12c4 to cuda/code/
- [ ] Verify file structure matches plan
- **Done when**: All core files copied and verified

## Task 1.2: Install Dependencies and Test CPU Pipeline
- [ ] Install warp-lang package
- [ ] Run ir_extractor.py with device="cpu" to verify baseline
- [ ] Run generator.py to verify kernel generation
- [ ] Run pipeline.py to generate 5 sample pairs
- **Done when**: CPU pipeline generates 5 valid pairs without errors

## Task 1.3: Analyze Warp CUDA Code Generation
- [ ] Study warp/context.py for device parameter handling
- [ ] Study warp/codegen.py for CUDA vs CPU differences
- [ ] Test ir_extractor with device="cuda" (will fail but observe error)
- [ ] Document CUDA code generation flow
- **Done when**: notes/cuda_analysis.md documents CUDA generation mechanism

## Task 1.4: Document Kernel Types
- [ ] List all kernel types from generator.py
- [ ] For each type, document key operations used
- [ ] Identify CUDA-specific considerations for each type
- **Done when**: notes/kernel_inventory.md lists all types with CUDA notes

## Task 1.5: CPU vs CUDA IR Comparison
- [ ] Generate sample CPU IR for each kernel type
- [ ] Study warp examples with CUDA output (from warp repo docs)
- [ ] Document key differences (headers, thread indexing, memory)
- **Done when**: notes/cpu_vs_cuda_ir.md documents differences with examples
