# Branch 3576 Analysis

## Quick Stats
- **Milestone**: M4 complete (per `README.md`)
- **Data generated**: 104 samples under `data/samples/`
- **Kernel categories**: 7 (includes `function` + `reduction`)

## Unique Features
- **Sample format split across files**:
  - `sample_XXXXX.py` (Python source)
  - `sample_XXXXX.cpp` (C++ IR)
  - `sample_XXXXX.json` (metadata)
  This is convenient for browsing/diffing and avoids huge JSON blobs.
- **Validation scripts**: `code/synthesis/validate_dataset.py` + `generate_dataset.py`.
- **“function” category**: includes `@wp.func` helpers (useful to broaden model coverage).
- **Extractor API**: README describes a structured `extract_ir()` result and `extract_ir_to_file()` helper.

## Recommended for Merge
- [ ] Adopt/offer the **3-file sample layout** as an optional output mode (or convert during export).
- [ ] Bring in generator templates for `function` + `reduction` if missing from the final base.
- [ ] Pull over validation scripts + dataset stats JSON generation ideas.

## Skip / Handle Carefully
- Contains many sample files; final merged repo should keep ≤100 samples.
- Exclude `__pycache__/` and `*.pyc`.

