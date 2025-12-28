# Tier 3-4 Branches Quick Scan

## Summary
Tier 3-4 branches are at M2-M3 milestone (IR extraction, no synthesis pipeline).
Most features are already covered by Tier 1-2 branches.

---

## 25e7 (Tier 3)
- **Features**: fast_generate.py, create_10k_dataset.py
- **Status**: M5 but only 9 data samples committed
- **Verdict**: SKIP - Utility scripts for scaling, not needed

## 5d09 (Tier 3) 
- **Features**: analyze_dataset.py, poisson_solver.py
- **Status**: M5, 0 data committed
- **Verdict**: SKIP - Similar utilities in 82cf

## a4fd (Tier 3)
- **Features**: example kernels (add, dot, saxpy)
- **Status**: M5, 1 sample
- **Verdict**: SKIP - Basic examples

## 0fbe (Tier 4)
- **Features**: fixture_kernels.py, Poisson solver with tests
- **Status**: M3
- **Verdict**: SKIP - Poisson solver in other branches

## 7288 (Tier 4)
- **Features**: Multiple example kernels, m2_generate_pairs.py
- **Status**: M3
- **Verdict**: SKIP - No synthesis pipeline

## 3f34 (Tier 4)
- **Features**: debug tools, debug_loop.py
- **Status**: M2
- **Verdict**: SKIP - Debug utilities only

## 4b76 (Tier 4)
- **Features**: Basic IR extractor with tests
- **Status**: M2
- **Verdict**: SKIP - IR extractor in 12c4

## d623 (Tier 4)
- **Features**: Categorized test cases:
  - case_arith.py, case_atomic.py, case_branch.py, case_loop.py, case_vec.py
- **Status**: M2
- **Verdict**: CONSIDER - Hand-written test cases by category could be useful for validation

### d623 Test Case Sample (case_arith.py)
```python
@wp.kernel
def k_arith(x, y, out):
    i = wp.tid()
    out[i] = (x[i] * 2.0 + y[i]) / 3.0
```

---

## Overall Verdict
- **SKIP ALL** except consider d623's categorized test cases for validation
- All key features are in Tier 1-2 branches
