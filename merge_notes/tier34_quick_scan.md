# Tier 3-4 Branches Quick Scan

## Tier 3: Partial Pipeline (M4-M5)

### Branch 25e7
- **Key files**: fast_generate.py, create_10k_dataset.py, generate_remaining.py
- **Value**: Fast generation scripts - utility scripts for scaling
- **Merge**: Fast generation approaches if different from main branches

### Branch 5d09
- **Key files**: analyze_dataset.py, example kernels (matrix_vector_mul.py, sine_wave.py)
- **Value**: Additional example kernels
- **Merge**: Example kernels if unique

### Branch a4fd
- **Key files**: Example kernels (add, dot, saxpy)
- **Data**: Only 1 sample
- **Value**: Classic HPC example kernels
- **Merge**: Example kernels (add.py, dot.py, saxpy.py) as reference implementations

## Tier 4: M2-M3 Only

### Branch 0fbe
- **Key files**: fixture_kernels.py, test_ir_extractor.py, poisson_solver.py
- **Value**: Fixture kernels for testing
- **Merge**: fixture_kernels.py if useful test fixtures

### Branch 7288
- **Key files**: m2_generate_pairs.py, example kernels (ex00_add.py, ex01_saxpy.py, ex02_reduction.py)
- **Value**: Classic example kernel collection
- **Merge**: Example kernels if well-documented

### Branch 3f34
- **Key files**: debug_loop.py, check_codegen.py, check_install.py
- **Value**: Debug and installation checking tools
- **Merge**: check_install.py could be useful

### Branch 4b76
- **Key files**: Basic IR extraction with tests
- **Value**: Minimal - covered by Tier 1 branches
- **Merge**: Skip - no unique features

### Branch d623
- **Key files**: Categorized test cases (case_arith.py, case_atomic.py, case_branch.py, case_loop.py, case_vec.py)
- **Value**: Well-organized test case categories
- **Merge**: Categorized test cases

## Summary
**Merge Priority (Tier 3-4)**:
1. **d623**: Categorized test cases
2. **a4fd/7288**: Classic HPC example kernels (add, saxpy, dot, reduction)
3. **25e7**: Fast generation utilities
4. **3f34**: check_install.py
5. **0fbe**: fixture_kernels.py if useful
6. Skip: 4b76 (no unique features), 5d09 (minimal unique value)
