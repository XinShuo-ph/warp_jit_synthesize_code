# Milestone 5 Tasks: Scale Up

## Task 1: Parallel Batch Generator
- [x] Step 1.1: Create `jit/code/synthesis/batch_generator.py`.
- [x] Step 1.2: Use `multiprocessing.Pool` to parallelize `synthesize_pair`.
- [x] Step 1.3: Ensure unique temporary filenames and thread-safe operations.
- **Done when**: `batch_generator.py` generates 100 samples significantly faster than sequential.

## Task 2: Large Scale Generation
- [x] Step 2.1: Run generation for 10,000 samples.
- [x] Step 2.2: Aggregate results into a single or partitioned JSONL files.
- **Done when**: `jit/data/` contains 10k+ valid pairs.

## Task 3: Dataset Statistics
- [x] Step 3.1: Analyze the dataset (token counts, operation types, average code length).
- [x] Step 3.2: Write `jit/notes/data_stats.md`.
- **Done when**: Statistics are documented.
