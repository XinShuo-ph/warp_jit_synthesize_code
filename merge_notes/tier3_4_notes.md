# Tier 3-4 Branches Quick Analysis

## Branch 25e7 (Tier 3 - Fast Generate)
- **Milestone**: M5 ✓
- **Files**: 31
- **Unique Feature**: `fast_generate.py` - Fast generation scripts
- **Verdict**: Check fast_generate.py for performance optimizations

## Branch 5d09 (Tier 3 - Analyze Dataset)
- **Milestone**: M5 ✓
- **Files**: 33
- **Unique Feature**: `analyze_dataset.py` - Dataset analysis tool
- **Verdict**: Similar to 82cf's analyze tool, check for differences

## Branch a4fd (Tier 3 - Example Kernels)
- **Milestone**: M5 ✓
- **Files**: 21
- **Data**: 1 sample (per branch_progresses.md: add, dot, saxpy examples)
- **Verdict**: Check for unique example kernels (add, dot, saxpy)

## Branch 0fbe (Tier 4 - M3)
- **Milestone**: M3 ✓ (Poisson solver)
- **Files**: 21
- **Unique Feature**: `fixture_kernels.py` - Test fixture kernels
- **Verdict**: Useful test fixtures for IR extraction

## Branch 7288 (Tier 4 - M3)
- **Milestone**: M3 ✓
- **Files**: 23
- **Features**: Poisson solver, example kernels (ex00_add.py, ex01_saxpy.py, ex02_reduction.py)
- **Verdict**: Good basic example kernels

## Branch 3f34 (Tier 4 - M2)
- **Milestone**: M2 ✓
- **Files**: 17
- **Unique Feature**: `debug_loop.py` - Debug tools
- **Verdict**: Useful debugging utilities

## Branch 4b76 (Tier 4 - M2)
- **Milestone**: M2 ✓ (basic)
- **Files**: 17
- **Verdict**: Basic IR extraction, likely superseded by better branches

## Branch d623 (Tier 4 - M2)
- **Milestone**: M2 ✓
- **Files**: 24
- **Unique Feature**: Categorized test cases in `cases/` directory
  - case_arith.py, case_atomic.py, case_branch.py, case_loop.py, case_vec.py
- **Verdict**: Well-organized test cases by category, good for test suite

## Recommended Merges from Tier 3-4
- ✅ **25e7**: fast_generate.py (if performance optimizations)
- ✅ **0fbe**: fixture_kernels.py (test fixtures)
- ✅ **7288**: Basic example kernels (add, saxpy, reduction)
- ✅ **3f34**: debug_loop.py (debugging)
- ✅ **d623**: Categorized test cases (case_*.py)
- ⚠️ **5d09**: analyze_dataset.py (compare with 82cf)
- ⚠️ **a4fd**: Check for unique examples
