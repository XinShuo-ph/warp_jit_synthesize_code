# CUDA Milestone 1 Tasks

## Task 1: Analyze Production Branches
- [ ] Step 1.1: List files from each `cursor/agent-work-merge-process-*` branch
- [ ] Step 1.2: Check for complete pipeline components (ir_extractor, generator, pipeline, batch_generator)
- [ ] Step 1.3: Compare code quality and completeness
- [ ] Step 1.4: Select best branch and document decision
- **Done when**: Best production branch identified and documented in CUDA_STATE.md

## Task 2: Reproduce CPU Pipeline
- [ ] Step 2.1: Copy selected branch's code to current branch
- [ ] Step 2.2: Install dependencies (warp-lang)
- [ ] Step 2.3: Run pipeline to generate 10+ CPU samples
- [ ] Step 2.4: Verify output format and quality
- **Done when**: CPU pipeline generates valid samples, documented in notes/cpu_baseline.md

## Task 3: Document CPU Baseline
- [ ] Step 3.1: Analyze CPU IR output format
- [ ] Step 3.2: Document kernel types supported
- [ ] Step 3.3: Document forward/backward pass structure
- [ ] Step 3.4: Save reference samples for comparison
- **Done when**: notes/cpu_baseline.md complete (max 50 lines), CPU samples saved
