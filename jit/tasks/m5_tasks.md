# Milestone 5 Tasks: Scale Up

## Task 1: Design Batch Generation System
- [ ] Step 1.1: Plan efficient batch generation (avoid memory issues)
- [ ] Step 1.2: Design chunked file saving strategy
- [ ] Step 1.3: Add progress tracking and checkpointing
- **Done when**: Clear architecture for large-scale generation

## Task 2: Implement Batch Generator
- [ ] Step 2.1: Create BatchGenerator class with chunking
- [ ] Step 2.2: Add configurable batch sizes and seeds
- [ ] Step 2.3: Implement progress saving/resuming
- [ ] Step 2.4: Add error recovery mechanisms
- **Done when**: batch_generator.py can generate thousands of pairs

## Task 3: Generate Large Dataset
- [ ] Step 3.1: Configure generation for 10k+ pairs
- [ ] Step 3.2: Run batch generation
- [ ] Step 3.3: Verify no duplicates or corrupted pairs
- [ ] Step 3.4: Save to data/ directory in chunks
- **Done when**: data/ contains 10k+ valid training pairs

## Task 4: Collect Dataset Statistics
- [ ] Step 4.1: Count total pairs by category
- [ ] Step 4.2: Calculate size statistics (Jaxpr length, StableHLO length)
- [ ] Step 4.3: Check diversity of operations
- [ ] Step 4.4: Create notes/data_stats.md (max 20 lines)
- **Done when**: notes/data_stats.md contains accurate statistics

## Task 5: Final Validation
- [ ] Step 5.1: Verify all pairs are loadable
- [ ] Step 5.2: Check JSON format consistency
- [ ] Step 5.3: Validate IR format integrity
- [ ] Step 5.4: Confirm dataset meets requirements
- **Done when**: Dataset is production-ready

## Success Criteria for M5
- [ ] code/synthesis/batch_generator.py exists and works
- [ ] data/ contains 10k+ Python→IR pairs
- [ ] notes/data_stats.md contains dataset statistics
- [ ] All pairs are valid and loadable
- [ ] Dataset has good diversity across categories
