# Tier 2 Branches Analysis (82cf, aa30, ff72, 3576, 3a5b)

## Branch 82cf (775 pairs)
- **Milestone**: M5 ✓
- **Data**: 775 pairs
- **Unique files**:
  - `code/synthesis/analyze_dataset.py` - Dataset statistics tool
  - `code/synthesis/validate_extraction.py` - Validation tools
  - `README.md`, `FINAL_REPORT.md`, `PROJECT_COMPLETE.md` - Documentation
- **Location**: Root level (no jit/ subdirectory)
- **Merge value**: Documentation files, analyze_dataset utility

## Branch aa30 (628 pairs)
- **Milestone**: M5 ✓
- **Data**: 628 pairs
- **Unique files**:
  - `QUICKSTART.md` - Quick start guide
  - `FINAL_REPORT.md`, `PROJECT_SUMMARY.md` - Reports
  - `code/examples/01_simple_kernel.py`, `02_vector_ops.py`, `03_control_flow.py` - Numbered examples
- **Location**: Root level (no jit/ subdirectory)
- **Merge value**: QUICKSTART guide, numbered example progression

## Branch ff72 (371 pairs)
- **Milestone**: M5 ✓
- **Data**: 266 pairs (data_stats says 266, branch_progresses says 371)
- **Generation**: Slow (~0.3-0.7 pairs/s)
- **7 kernel types**: arithmetic, math, loop, conditional, vector, matrix, combined
- **Location**: `jit/` subdirectory
- **Merge value**: Similar to 12c4, no clear advantage

## Branch 3576 (239 pairs)
- **Milestone**: M4 ✓
- **Data**: 239 .py files
- **Unique files**:
  - `code/synthesis/validate_dataset.py` - Dataset validation
  - `data/test_cases/test_arithmetic.py`, `test_control_flow.py`, `test_functions.py` - Categorized test cases
- **Location**: Root level
- **Merge value**: Validation script, categorized test structure

## Branch 3a5b (100 pairs)
- **Milestone**: M5 ✓
- **Data**: 100 pairs
- **Location**: `jit/` subdirectory (330 files total)
- **Merge value**: Small dataset, likely similar to others

## Summary - Tier 2 Recommendations

### High value for merge:
1. **82cf**: `analyze_dataset.py` - useful analytics
2. **aa30**: `QUICKSTART.md` - good user onboarding
3. **3576**: `validate_dataset.py` - quality checks

### Skip:
- ff72, 3a5b: Similar functionality to 12c4 but smaller/slower

### Note:
Most Tier 2 branches use **root-level** code structure (no jit/ subdirectory), while 12c4/9177/8631 use `jit/` subdirectory. Will need to handle this during merge.
