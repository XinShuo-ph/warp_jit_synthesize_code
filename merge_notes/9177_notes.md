# Branch 9177 Analysis

## Quick Stats
- **Milestone**: M5 complete (per `jit/README.md`)
- **Data generated**: 10,270 pairs (10,150 training + 120 samples)
- **Pipeline works**: Yes (CPU-only; ran `--count 1`)

## What I Verified
- Ran: `python3 /tmp/test_9177/code/synthesis/pipeline.py --count 1 --output /tmp/test_9177_out`
- Successfully generated 1 pair (type `arithmetic`), with Warp compiling a temp module on CPU.

## Unique Features (vs 12c4)
- **10 kernel types** (broader than 12c4’s documented 6–7):
  - arithmetic, conditional, loop, math, vector, atomic, nested, multi_cond, combined, scalar_param
- **Forward + backward IR extraction**:
  - `pipeline.py` extracts both `*_kernel_forward` and `*_kernel_backward` functions when available.
- **Stable pair IDs and richer metadata**:
  - `id = sha256(python_source)[:12]`
  - stores `module_id`, `num_lines`, `num_params`
- **IR extraction via warp cache**:
  - reads `~/.cache/warp/<version>/<module_id>/<module_id>.cpp` after compilation instead of invoking Warp codegen directly.

## Code Quality
- **Clean**: Generally yes; still contains `__pycache__/` artifacts in repo.
- **Tests**: Has `jit/code/extraction/test_ir_extractor.py`
- **Docs**: Clear README + notes, includes dataset stats and kernel type taxonomy.

## Recommended for Merge
- [ ] `jit/code/synthesis/generator.py` - consider adopting for expanded kernel-type coverage (10 categories).
- [ ] `jit/code/synthesis/pipeline.py` - consider adopting pair ID scheme + optional backward IR extraction approach.

## Skip / Handle Carefully
- **Large dataset**: `jit/data/training/` is 10k+ files; final merged repo should keep ≤100 samples.
- **Bytecode artifacts**: `jit/**/__pycache__/*` and `*.pyc` should be excluded.

