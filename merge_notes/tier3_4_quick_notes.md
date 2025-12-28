# Tier 3 & 4 Branches Quick Scan

## Tier 3 (Pipeline Started)

### Branch 25e7 (9 samples, fast_generate scripts)
- **Files**: fast_generate.py, create_10k_dataset.py, generate_remaining.py
- **Milestone**: M5
- **Unique**: Fast generation scripts, automation tools
- **Data**: Only 9 samples committed
- **Verdict**: **SKIP** - Other branches have more complete solutions

### Branch 5d09 (0 committed, pipeline complete)
- **Files**: analyze_dataset.py, pipeline complete
- **Milestone**: M5
- **Unique**: Examples (poisson_solver, matrix_vector_mul, sine_wave)
- **Data**: 0 samples in repo
- **Verdict**: **SKIP** - No data, similar to other branches

### Branch a4fd (1 sample, example kernels)
- **Files**: Example kernels (add, dot, saxpy)
- **Milestone**: M5
- **Unique**: Clean kernel examples
- **Data**: 1 sample
- **Verdict**: **MAYBE** - Good simple examples, but similar to others

---

## Tier 4 (M2-M3 Only)

### Branch 0fbe (M3, fixture_kernels)
- **Files**: fixture_kernels.py, ir_extractor, poisson_solver
- **Milestone**: M3 (Poisson solver)
- **Unique**: Fixture kernels for testing
- **Data**: Limited
- **Verdict**: **MAYBE** - fixture_kernels.py could be useful for tests

### Branch 7288 (M3, example kernels)
- **Files**: m2_generate_pairs.py, ex00_add, ex01_saxpy, ex02_reduction
- **Milestone**: M3
- **Unique**: Numbered example progression
- **Data**: Limited
- **Verdict**: **SKIP** - Similar to ff72 examples

### Branch 3f34 (M2, debug tools)
- **Files**: debug_loop.py, check_codegen.py, check_install.py
- **Milestone**: M2 (IR extraction)
- **Unique**: Debug utilities, installation checks
- **Verdict**: **MAYBE** - Debug tools could be useful

### Branch 4b76 (M2, basic extraction)
- **Files**: ir_extractor.py, test_ir_extractor.py
- **Milestone**: M2
- **Unique**: Nothing unique
- **Verdict**: **SKIP** - Basic M2 implementation

### Branch d623 (M2, categorized test cases)
- **Files**: test_cases.py, cases/ directory (case_arith, case_atomic, case_branch, case_loop, case_vec)
- **Milestone**: M2
- **Unique**: **Highly organized test cases by category**
- **Verdict**: **YES** - Excellent test organization, merge the categorized test cases

---

## Summary: Tier 3-4 Recommendations

### Must Merge:
- **d623**: Categorized test cases (case_arith.py, case_atomic.py, etc.)

### Consider Merging:
- **0fbe**: fixture_kernels.py
- **3f34**: debug_loop.py, check_codegen.py, check_install.py

### Skip:
- 25e7, 5d09, a4fd, 7288, 4b76 - No unique valuable features
