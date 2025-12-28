# Tier 3-4 Branches Analysis (25e7, 5d09, a4fd, 0fbe, 7288, 3f34, 4b76, d623)

## Tier 3: Pipeline Started (M4-M5)

### Branch 25e7 (9 pairs) - M5 ✓
- **Data**: 9 pairs only
- **Unique files**: `fast_generate.py`, `create_10k_dataset.py`, `generate_remaining.py` scripts
- **Location**: Root level (code/)
- **Merge value**: Script ideas for fast generation (but 12c4 already faster)

### Branch 5d09 (0 pairs committed) - M5 ✓
- **Data**: 0 pairs in repo (pipeline complete but not committed)
- **Unique files**: 
  - `code/synthesis/analyze_dataset.py`
  - Example kernels: `matrix_vector_mul.py`, `sine_wave.py`
- **Location**: jit/ subdirectory
- **Merge value**: Example kernels

### Branch a4fd (1 pair) - M5 ✓
- **Data**: 1 pair
- **Example kernels**: `test_add_kernel.py`, `test_dot_product.py`, `test_saxpy.py`
- **Location**: jit/ subdirectory
- **Merge value**: Test kernel examples

## Tier 4: M2-M3 Complete (Poisson Solver / IR Extraction)

### Branch 0fbe (M3 ✓) - Poisson solver + IR extraction
- **Unique files**: 
  - `code/extraction/fixture_kernels.py` - Collection of test kernels (add_constant, conditional_scale, struct_math, atomic_accumulate, trig_mix)
- **Location**: jit/ subdirectory
- **Merge value**: **High** - fixture_kernels.py provides diverse test cases

### Branch 7288 (M3 ✓) - Poisson solver + example kernels
- **Unique files**:
  - `code/examples/ex00_add.py`, `ex01_saxpy.py`, `ex02_reduction.py` - Numbered progression
  - `code/extraction/m2_generate_pairs.py` - IR pair generation script
- **Location**: jit/ subdirectory
- **Merge value**: Numbered example progression

### Branch 3f34 (M2 ✓) - IR extractor + debug tools
- **Unique files**:
  - `code/extraction/debug_loop.py` - Debug utilities
  - `code/examples/check_codegen.py`, `check_install.py` - Setup validation
- **Location**: Root level (code/)
- **Merge value**: Debug and validation utilities

### Branch 4b76 (M2 ✓) - IR extractor with tests
- **Files**: Basic IR extractor + tests
- **Location**: jit/ subdirectory
- **Merge value**: Low - similar to 12c4

### Branch d623 (M2 ✓) - IR extractor with categorized test cases
- **Unique files**:
  - `code/extraction/cases/` directory with categorized test kernels:
    - `case_arith.py` - Arithmetic operations
    - `case_atomic.py` - Atomic operations
    - `case_branch.py` - Conditionals
    - `case_loop.py` - Loops
    - `case_vec.py` - Vector operations
- **Location**: jit/ subdirectory
- **Merge value**: **High** - categorized test case structure

## Summary - Tier 3-4 Recommendations

### High value for merge:
1. **0fbe**: `fixture_kernels.py` - diverse test kernels (add_constant, conditional_scale, struct_math, atomic_accumulate, trig_mix)
2. **d623**: `code/extraction/cases/` - categorized test case structure (arith, atomic, branch, loop, vec)
3. **3f34**: Debug utilities (`debug_loop.py`, `check_codegen.py`, `check_install.py`)

### Medium value:
- **7288**: Numbered example progression (ex00, ex01, ex02...)
- **5d09**: Example kernels (matrix_vector_mul, sine_wave)

### Skip:
- 25e7, a4fd, 4b76: Minimal unique value

### Note:
Most useful contributions from Tier 3-4 are **test fixtures and categorized test cases**, not large datasets.
