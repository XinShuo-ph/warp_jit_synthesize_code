# Milestone 5 Tasks

## Task 1: Batch Generator
- [ ] Step 1.1: Create `code/synthesis/batch_generator.py`.
- [ ] Step 1.2: Implement multiprocessing support to run synthesis in parallel.
- [ ] Step 1.3: Add progress tracking (tqdm or simple print).
- **Done when**: `batch_generator.py` can generate 1k samples faster than serial pipeline.

## Task 2: Large Scale Generation
- [ ] Step 2.1: Run batch generator to create 10k samples.
- [ ] Step 2.2: Ensure data is saved efficiently (e.g., one JSONL file vs 10k small JSON files).
- [ ] Step 2.3: Compute basic stats (avg IR length, strategy distribution).
- **Done when**: `data/dataset.jsonl` exists with 10k+ lines.

## Task 3: Final Documentation
- [ ] Step 3.1: Create `notes/data_stats.md`.
- [ ] Step 3.2: Clean up temporary files.
- **Done when**: Project is clean and documented.
