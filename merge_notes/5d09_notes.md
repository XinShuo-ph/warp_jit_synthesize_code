# Branch 5d09 Analysis

## Quick Stats
- **Milestone**: M5 complete (per `jit/README.md`)
- **Data generated**: 10,000 samples in JSONL (`jit/data/large_scale/dataset_large.jsonl`)
- **Output content**: includes both **forward + backward** C++ snippets per sample

## Unique Features
- **JSONL dataset format** (streamable) with forward/backward fields:
  - `cpp_source_forward`, `cpp_source_backward`
- **Dataset analysis script**: `jit/code/synthesis/analyze_dataset.py`
- **Generator focus**: random syntactically valid kernels including casts/array access (per README)

## Recommended for Merge
- [ ] Consider adopting the **JSONL schema** (especially if we want forward/backward pairs).
- [ ] Mine `analyze_dataset.py` ideas if we want richer reporting.

## Skip / Handle Carefully
- **Large committed dataset** (`dataset_large.jsonl`) should not be included in final merged repo; keep ≤100 samples.
- Temp modules under `jit/code/synthesis/temp/` and `__pycache__/` should not be merged.

