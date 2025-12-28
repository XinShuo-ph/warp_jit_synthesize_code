# Tier 4 Branches Analysis (0fbe, 7288, 3f34, 4b76, d623)

## 0fbe Analysis
- Milestone: M3 ✓
- Pipeline: N/A (M3 only)
- **fixture_kernels.py: TESTED WORKING**
```python
import fixture_kernels
# Kernels: ['Pair', 'add_constant', 'atomic_accumulate', 
#           'conditional_scale', 'struct_math', 'trig_mix', 'wp']
```
- Includes 5 test kernels + Pair struct
- Useful for testing

## 7288 Analysis
- Milestone: M3 ✓
- Has numbered examples (ex00_add, ex01_saxpy, ex02_reduction)
- No pipeline
- Skip: M3 only, limited scope

## 3f34 Analysis
- Milestone: M2 ✓
- Has debug_loop.py, test_cuda_codegen.py
- No pipeline
- Skip: M2 only, debugging tools

## 4b76 Analysis
- Milestone: M2 ✓
- Has basic IR extraction
- No pipeline
- Skip: M2 only, basic extraction

## d623 Analysis
- Milestone: M2 ✓
- **Categorized test cases: TESTED WORKING**
```python
import case_arith
kernel = case_arith.get_kernel()  # Works!
```
- cases/: case_arith.py, case_atomic.py, case_branch.py, case_loop.py, case_vec.py
- Useful for testing

## Recommended for Merge
- [x] 0fbe: fixture_kernels.py - 5 test kernels + struct
- [x] d623: cases/*.py - Categorized test cases

## Skip
- 7288, 3f34, 4b76: M2-M3 only, no unique features
