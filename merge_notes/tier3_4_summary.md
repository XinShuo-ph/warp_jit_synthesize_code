# Tier 3-4 Branches Summary - Quick Tests

## Tier 3: Pipeline Started (M4-M5)

### 25e7
- Has pipeline.py but requires module path fixes
- Has fast_generate.py, create_10k_dataset.py
- **Test result**: ModuleNotFoundError (import issues)
- SKIP: Import path problems

### 5d09
- Has pipeline.py 
- Has analyze_dataset.py
- **Test result**: Not fully tested (similar import issues)
- SKIP: Similar to 12c4

### a4fd
- Has pipeline.py
- Has simple test kernels
- **Test result**: Not fully tested
- SKIP: Basic functionality only

## Tier 4: M3 Complete (No synthesis pipeline)

### 0fbe, 7288, 3f34, 4b76, d623
- **No pipeline.py** - M2/M3 only (IR extraction, no synthesis)
- Have ir_extractor.py and test kernels
- SKIP: No synthesis capability

## Summary
- Only Tier 1-2 branches have working synthesis pipelines
- Tier 3 branches have import issues
- Tier 4 branches lack synthesis pipelines entirely

## Final Recommendation
- **Primary base**: 12c4 (6 types, clean, documented)
- **Merge from**: 
  - ff72: combined kernel type (7th type)
  - 9177: Consider additional types (nested_loop, multi_conditional)
- **Skip**: All Tier 3-4 branches
