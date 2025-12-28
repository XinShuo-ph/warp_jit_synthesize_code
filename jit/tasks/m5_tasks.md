# Milestone 5 Tasks: Scale Up

## Task 1: Create Batch Generator
- [x] Step 1.1: Create batch_generator.py with batched module compilation
- [x] Step 1.2: Implement chunked generation to avoid memory issues
- [x] Step 1.3: Add progress tracking
- **Done when**: Can generate 1000+ pairs efficiently ✓ (~180 pairs/sec)

## Task 2: Generate Large Dataset
- [x] Step 2.1: Generate 10k+ varied kernel pairs (10,500 total)
- [x] Step 2.2: Ensure diversity across all categories (balanced ~17% each)
- [x] Step 2.3: Save to data/large/ directory
- **Done when**: data/ contains 10k+ validated pairs ✓

## Task 3: Document Dataset Statistics
- [x] Step 3.1: Compute statistics (category distribution, size)
- [x] Step 3.2: Create notes/data_stats.md with summary
- **Done when**: notes/data_stats.md exists with <20 lines ✓

## Optimization Strategies
1. Batch kernel generation (avoid module reloading overhead)
2. Generate multiple kernels per module file
3. Skip backward codegen for faster compilation
4. Use process pools for parallel compilation
