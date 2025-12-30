# JIT Branch Progress Summary

## Overview
16 branches working on Python→IR code synthesis for LLM training data using JAX.

## Milestone Reference
- **M1**: Environment setup, run JAX examples
- **M2**: IR extraction mechanism
- **M3**: Numerical PDE deep dive, Poisson solver
- **M4**: Synthesis pipeline
- **M5**: Scale up (batch generation)

---

## Branch Details

### Tier 1: Production Ready (M5 Complete, Large Dataset)

| Branch | Milestone | Data Count | Key Features |
|--------|-----------|------------|--------------|
| **12c4** | M5 ✓ | **10,727** | Full pipeline, batch generator, all 5 task files |
| **9177** | M5 ✓ | **10,320** | Complete project, training data generated |
| **8631** | M4 ✓ | **10,101** | IR extraction + synthesis pipeline |

**12c4 Files:**
- `code/extraction/ir_extractor.py`, `save_sample_pairs.py`
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`
- `code/examples/poisson_solver.py`, `test_poisson.py`
- `notes/data_stats.md`, `ir_format.md`, `jax_basics.md`

**9177 Files:**
- `code/extraction/ir_extractor.py`, `test_ir_extractor.py`
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`
- `notes/data_stats.md`, `ir_format.md`

**8631 Files:**
- `code/extraction/ir_extractor.py`, `debug_extraction.py`
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`
- `code/examples/poisson_solver.py`, example kernels

---

### Tier 2: Complete Pipeline (M4-M5, Smaller Dataset)

| Branch | Milestone | Data Count | Key Features |
|--------|-----------|------------|--------------|
| **82cf** | M5 ✓ | **775** | Complete project, validation scripts, README |
| **aa30** | M5 ✓ | **628** | Full pipeline, FINAL_REPORT, QUICKSTART |
| **ff72** | M5 ✓ | **371** | Clean documentation, all task files |
| **3576** | M4 ✓ | **239** (.py) | Dataset validation, test cases by category |
| **3a5b** | M5 ✓ | **100** | Kernel generation, batch generator |

**82cf Files:**
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`, `analyze_dataset.py`
- `code/extraction/ir_extractor.py`, `validate_extraction.py`
- `README.md`, `FINAL_REPORT.md`, `PROJECT_COMPLETE.md`

**aa30 Files:**
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`
- `code/examples/01_simple_kernel.py` through `03_control_flow.py`
- `README.md`, `QUICKSTART.md`, `FINAL_REPORT.md`

**ff72 Files:**
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`
- `code/examples/ex1_basic_kernel.py`, `ex2_math_ops.py`, `ex3_vec_types.py`
- `code/examples/poisson_solver.py`, `test_poisson.py`
- All 5 milestone task files

**3576 Files:**
- `code/synthesis/generator.py`, `pipeline.py`, `generate_dataset.py`, `validate_dataset.py`
- `data/test_cases/test_arithmetic.py`, `test_control_flow.py`, `test_functions.py`, etc.
- 100+ sample files in `data/samples/`

---

### Tier 3: Pipeline Started (M4)

| Branch | Milestone | Data Count | Key Features |
|--------|-----------|------------|--------------|
| **25e7** | M5 ✓ | 9 | Fast generate, create_10k scripts |
| **5d09** | M5 ✓ | 0 | Pipeline complete, no data committed |
| **a4fd** | M5 ✓ | 1 | Example kernels (add, dot, saxpy) |

**25e7 Files:**
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`, `fast_generate.py`
- `create_10k_dataset.py`, `generate_remaining.py`
- All 5 milestone task files

**5d09 Files:**
- `code/synthesis/generator.py`, `pipeline.py`, `batch_generator.py`, `analyze_dataset.py`
- `code/examples/poisson_solver.py`, `matrix_vector_mul.py`, `sine_wave.py`
- All 5 milestone task files

---

### Tier 4: M3 Complete (Poisson Solver)

| Branch | Milestone | Key Features |
|--------|-----------|--------------|
| **0fbe** | M3 ✓ | Poisson solver with tests, IR extraction, fixture kernels |
| **7288** | M3 ✓ | Poisson solver, multiple example kernels, IR pair generation |

**0fbe Files:**
- `code/extraction/ir_extractor.py`, `test_ir_extractor.py`, `fixture_kernels.py`
- `code/examples/poisson_solver.py`, `test_poisson.py`
- `notes/ir_format.md`, `jax_basics.md`

**7288 Files:**
- `code/extraction/ir_extractor.py`, `m2_generate_pairs.py`
- `code/examples/poisson_solver.py`, `ex00_add.py`, `ex01_saxpy.py`, `ex02_reduction.py`

---

### Tier 5: M2 Complete (IR Extraction)

| Branch | Milestone | Key Features |
|--------|-----------|--------------|
| **3f34** | M2 ✓ | IR extractor, debug tools |
| **4b76** | M2 ✓ | IR extractor with tests |
| **d623** | M2 ✓ | IR extractor with categorized test cases |

**3f34 Files:**
- `code/extraction/ir_extractor.py`, `test_extractor.py`, `debug_loop.py`
- `code/examples/check_codegen.py`, `check_install.py`

**4b76 Files:**
- `code/extraction/ir_extractor.py`, `test_ir_extractor.py`
- `notes/ir_format.md`, `jax_basics.md`

**d623 Files:**
- `code/extraction/ir_extractor.py`, `test_cases.py`
- `code/extraction/cases/case_arith.py`, `case_atomic.py`, `case_branch.py`, `case_loop.py`, `case_vec.py`

---

## Recommended Merge Strategy

### For Code:
1. **Primary base**: `12c4` (most complete, largest dataset)
2. **Documentation**: `82cf` or `aa30` (has README, reports)
3. **Test cases**: `d623` (categorized test cases), `3576` (validation by category)

### For Data:
- `12c4`: 10,727 JSON pairs
- `9177`: 10,320 JSON pairs  
- `8631`: 10,101 JSON pairs
- Total unique: ~30k+ (need dedup)

### Key Components to Merge:
| Component | Best Source |
|-----------|-------------|
| `ir_extractor.py` | 12c4, ff72 |
| `generator.py` | 12c4, ff72 (7 kernel types) |
| `pipeline.py` | 12c4, 82cf |
| `batch_generator.py` | 12c4, 9177 |
| `poisson_solver.py` | 0fbe, ff72 |
| Test cases | d623, 3576 |
| README | 82cf, aa30 |
| Data samples | 12c4, 9177, 8631 |

---

## Common File Structure

Most complete branches follow:
```
jit/
├── code/
│   ├── examples/         # JAX example programs
│   ├── extraction/       # ir_extractor.py + tests
│   └── synthesis/        # generator.py, pipeline.py, batch_generator.py
├── data/                 # Generated JSON pairs
├── notes/                # ir_format.md, jax_basics.md, data_stats.md
└── tasks/                # m1_tasks.md through m5_tasks.md
```

