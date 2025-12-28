# Tier 3-4 Branches Quick Scan

## Tier 3: Pipeline Started (M4+)

### 25e7
- Milestone: M5
- Data: 9 samples only
- Key features: fast_generate.py, create_10k_dataset.py scripts
- Status: Skip - minimal data, 12c4 is better

### 5d09
- Milestone: M5
- Data: 0 committed
- Key features: analyze_dataset.py, complete pipeline
- Status: Skip - no data, similar to others

### a4fd
- Milestone: M5
- Data: 1 sample
- Key features: test_add_kernel.py, test_dot_product.py, test_saxpy.py examples
- Status: Skip - minimal content

---

## Tier 4: M2-M3 Only

### 0fbe
- Milestone: M3
- Key features: **fixture_kernels.py** with varied test kernels (add_constant, conditional_scale, struct_math, atomic_accumulate, trig_mix)
- Status: Partial merge - fixture_kernels.py is useful for tests

### 7288
- Milestone: M3
- Key features: Well-numbered examples (ex00_add, ex00_smoke, ex01_saxpy, ex02_reduction)
- Status: Skip - similar to other examples

### 3f34
- Milestone: M2
- Key features: debug_loop.py, check_codegen.py, test_cuda_codegen.py
- Status: Skip - debugging tools, not production

### 4b76
- Milestone: M2
- Key features: Basic IR extractor with tests
- Status: Skip - 12c4 is more complete

### d623
- Milestone: M2
- Key features: **Categorized test cases** in cases/ directory (case_arith.py, case_atomic.py, case_branch.py, case_loop.py, case_vec.py)
- Status: Partial merge - categorized cases could be useful for tests

---

## Recommended Merges from Tier 3-4

1. **0fbe/fixture_kernels.py** - Diverse test kernels with struct usage
2. **d623/cases/** - Categorized test cases

## Skip
All other code - 12c4 and ff72 cover the functionality better.
