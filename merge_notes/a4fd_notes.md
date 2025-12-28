# Branch a4fd Analysis

## Quick Stats
- **Milestone**: M5 complete (per `jit/README.md`)
- **Data generated**: 1,505 pairs in JSONL (`jit/data/training_all.jsonl`, ~8.2MB)
- **Kernel templates**: 10 categories (elementwise → compound)

## Unique Features
- **JSONL output first-class**: `pipeline.py` supports `--jsonl` and custom output path; `batch_generator.py` supports checkpointing/resume.
- **10 generator templates**: includes `reduction`, `nested_branch`, `multi_statement`, etc. (good diversity).
- **Small example tests** under `jit/code/examples/` (add/dot/saxpy) in addition to extraction tests.

## Recommended for Merge
- [ ] Mine generator templates for broader kernel variety (10 categories).
- [ ] Mine JSONL + checkpoint patterns (CLI flags, checkpoint file handling).

## Skip / Handle Carefully
- `training_all.jsonl` and other bulk datasets should not be merged into the final repo; keep ≤100 samples.
- Exclude `__pycache__/` and generated artifacts.

