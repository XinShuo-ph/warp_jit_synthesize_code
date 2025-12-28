# Branch 8631 Analysis

## Quick Stats
- Milestone: **M4** (pipeline + extraction), but large data reported (**~10,101** pairs)
- Pipeline works: Likely yes

## Unique Features
- **Debug tooling**: `code/extraction/debug_extraction.py`
- **Complete synthesis stack**: `code/synthesis/{generator,pipeline,batch_generator}.py`
- **Examples**: Poisson + example kernels

## Recommended for Merge
- [ ] `debug_extraction.py` - keep as optional debug utility (if clean/non-invasive)
- [ ] Compare `ir_extractor.py` robustness vs 12c4

## Skip
- Bulk dataset (keep small curated sample only).

