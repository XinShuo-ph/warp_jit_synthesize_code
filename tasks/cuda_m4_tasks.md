# Milestone 4 Tasks: CUDA IR Production Pipeline

## Task 1: Production Script
- [ ] Step 1.1: Create `code/synthesis/produce_ir.py`.
- [ ] Step 1.2: Ensure it sets up Warp for offline CUDA generation (no driver init checks if possible, or handling warnings).
- [ ] Step 1.3: Add regex validation for CUDA keywords in output.
- **Done when**: `python produce_ir.py --count 10` runs and generates valid files.

## Task 2: Mass Production
- [ ] Step 2.1: Run generation for 100 pairs.
- [ ] Step 2.2: Verify a random sample.
- **Done when**: `data/cuda_v1/` contains 100+ JSONs.

## Task 3: Documentation
- [ ] Step 3.1: Create `notes/cuda_production_notes.md`.
- [ ] Step 3.2: Document the offline generation capabilities and limitations (no PTX without NVCC).
- **Done when**: Notes exist.
